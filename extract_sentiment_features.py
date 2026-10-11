# extract frozen LM features for the labeled IMDb sentiment task
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from lm_data import PAD, UNK, BOS, EOS
from tokenize_imdb import tokenize
from lstm_model import LSTMLanguageModel
from transformer_model import TransformerLanguageModel


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def prepare_documents(rows, data_dir, word_to_id):
    started = time.perf_counter()
    documents = []
    unknown_count = 0
    token_count = 0
    for row in rows:
        path = data_dir / row['review_id']
        if file_hash(path) != row['sha256']:
            raise ValueError(f'Review changed since the split was created: {path}')
        tokens = tokenize(path.read_text(encoding='utf-8'))
        if not tokens:
            raise ValueError(f'Empty tokenized review: {path}')
        ids = [word_to_id.get(token, UNK) for token in tokens]
        token_count += len(ids)
        unknown_count += ids.count(UNK)
        # BOS appears once at the review start and EOS is not needed for classification
        documents.append(array('i', [BOS] + ids))
    statistics = {'documents': len(rows), 'tokens': token_count,
                  'unknown_tokens': unknown_count, 'unk_rate': unknown_count / token_count,
                  'preparation_seconds': time.perf_counter() - started}
    return documents, statistics


def make_inputs(sequences, device):
    width = max(len(sequence) for sequence in sequences)
    inputs = torch.full((len(sequences), width), PAD, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        inputs[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return inputs.to(device)


def accumulate_states(states, valid, document_indices, sums, maxima, counts):
    batch_sums = states.masked_fill(~valid.unsqueeze(-1), 0).sum(dim=1).cpu().numpy()
    batch_maxima = states.masked_fill(~valid.unsqueeze(-1), float('-inf')).amax(dim=1).cpu().numpy()
    batch_counts = valid.sum(dim=1).cpu().numpy()
    for index, document_index in enumerate(document_indices):
        sums[document_index] += batch_sums[index]
        maxima[document_index] = np.maximum(maxima[document_index], batch_maxima[index])
        counts[document_index] += batch_counts[index]


@torch.inference_mode()
def extract_lstm(model, documents, args, device, sums, maxima, counts):
    order = sorted(range(len(documents)), key=lambda index: len(documents[index]), reverse=True)
    chunk_count = 0
    for offset in range(0, len(order), args.batch_size):
        group = order[offset:offset + args.batch_size]
        hidden = None
        longest = len(documents[group[0]])
        for start in range(0, longest, args.chunk_length):
            active = [index for index in group if len(documents[index]) > start]
            sequences = [documents[index][start:start + args.chunk_length] for index in active]
            inputs = make_inputs(sequences, device)
            if hidden is not None:
                # finished reviews leave the batch. active reviews form a prefix
                hidden = (hidden[0][:, :len(active)].contiguous(),
                          hidden[1][:, :len(active)].contiguous())
            states, hidden = model.lstm(model.embedding(inputs), hidden)
            valid = (inputs != PAD) & (inputs != BOS) & (inputs != EOS)
            accumulate_states(states, valid, active, sums, maxima, counts)
            chunk_count += len(active)
        # a fresh group starts with hidden=None preventing cross-review leakage
        if offset // args.batch_size % 50 == 0:
            print(f'  reviews {min(offset + args.batch_size, len(order))}/{len(order)}', flush=True)
    return chunk_count


@torch.inference_mode()
def extract_transformer(model, documents, args, device, sums, maxima, counts):
    windows = []
    for document_index, ids in enumerate(documents):
        start = 0
        next_unpooled = 0
        while next_unpooled < len(ids):
            stop = min(start + args.chunk_length, len(ids))
            # overlap supplies context but not contribute duplicate states
            keep_from = next_unpooled - start
            windows.append((document_index, start, stop, keep_from))
            next_unpooled = stop
            start += args.stride
    for offset in range(0, len(windows), args.batch_size):
        batch = windows[offset:offset + args.batch_size]
        sequences = [documents[index][start:stop] for index, start, stop, keep in batch]
        inputs = make_inputs(sequences, device)
        states = model.token_states(inputs)
        valid = (inputs != PAD) & (inputs != BOS) & (inputs != EOS)
        for index, (_, _, _, keep_from) in enumerate(batch):
            valid[index, :keep_from] = False
        document_indices = [window[0] for window in batch]
        accumulate_states(states, valid, document_indices, sums, maxima, counts)
        if offset // args.batch_size % 200 == 0:
            print(f'  windows {min(offset + args.batch_size, len(windows))}/{len(windows)}', flush=True)
    return len(windows)


def extract_split(model, rows, data_dir, word_to_id, args, device):
    documents, statistics = prepare_documents(rows, data_dir, word_to_id)
    if args.model == 'lstm':
        dimension = model.lstm.hidden_size
    else:
        dimension = model.input_projection.out_features
    sums = np.zeros((len(rows), dimension), dtype=np.float64)
    maxima = np.full((len(rows), dimension), -np.inf, dtype=np.float32)
    counts = np.zeros(len(rows), dtype=np.int64)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    if args.model == 'lstm':
        chunk_count = extract_lstm(model, documents, args, device, sums, maxima, counts)
    else:
        chunk_count = extract_transformer(model, documents, args, device, sums, maxima, counts)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    extraction_seconds = time.perf_counter() - started
    mean_features = (sums / counts[:, None]).astype(np.float32)
    features = np.concatenate([mean_features, maxima], axis=1)
    labels = np.array([row['label'] for row in rows], dtype=np.int64)
    review_ids = np.array([row['review_id'] for row in rows])
    statistics['chunks_or_windows'] = chunk_count
    statistics['extraction_seconds'] = extraction_seconds
    statistics['extraction_documents_per_second'] = len(rows) / extraction_seconds
    return features, labels, review_ids, counts, statistics


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['lstm', 'transformer'])
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=base / 'sentiment_splits')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--chunk-length', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64,
                        help='Transformer window stride; ignored for LSTM.')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.chunk_length, args.cpu_threads) < 1:
        parser.error('Batch size, chunk length and CPU threads must be positive.')
    if args.model == 'transformer' and not 0 < args.stride < args.chunk_length:
        parser.error('Transformer stride must be positive and smaller than chunk length.')
    source = args.results_dir.resolve()
    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / 'manifest.json'
    if not manifest_path.is_file():
        parser.error('Prepared dataset missing. Run split_sentiment_data.py first, or set --data-dir to its output.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('dataset_format') != 'imdb_sentiment_txt_v1':
        parser.error('Unsupported dataset manifest; use the output of split_sentiment_data.py.')
    output = args.output_dir or source / 'sentiment_features'
    if output.exists():
        parser.error(f'Output already exists: {output}. Choose a fresh --output-dir.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use your training environment or --device cpu.')
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checkpoint_path = source / 'best_model.pt'
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if args.chunk_length > saved['config']['context_length']:
        parser.error('Chunk length exceeds the LM training context length.')
    if args.model == 'lstm':
        model = LSTMLanguageModel(**saved['model_config'])
    else:
        model = TransformerLanguageModel(**saved['model_config'])
    model.load_state_dict(saved['model_state'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device).eval()
    word_to_id = {word: index for index, word in enumerate(saved['vocabulary'])}
    output.mkdir(parents=True)
    if args.model == 'lstm':
        context_policy = 'carry hidden and cell states within each review; reset between reviews'
    else:
        context_policy = 'overlapping windows; reset positions; pool each token once from its first covering window'
    metadata = {'feature_schema_version': 2,
                'representation': source.name, 'model': args.model,
                'checkpoint': str(checkpoint_path), 'checkpoint_sha256': file_hash(checkpoint_path),
                'model_config': saved['model_config'], 'vocab_sha256': saved['vocab_sha256'],
                'manifest': str(manifest_path),
                'manifest_sha256': file_hash(manifest_path), 'data_dir': str(data_dir),
                'dataset_split_seed': manifest['split_seed'],
                'pooling_exclusions': ['PAD', 'BOS', 'EOS'],
                'chunk_length': args.chunk_length, 'context_policy': context_policy,
                'stride': args.stride if args.model == 'transformer' else None,
                'pooling': 'mean_max',
                'feature_dimension': 2 * model.output.in_features,
                'truncation': 'none; use every review token', 'language_model_frozen': True,
                'device': str(device), 'batch_size': args.batch_size,
                'versions': {'torch': str(torch.__version__), 'numpy': np.__version__},
                'timing_scope': 'extraction includes batching, transfers, hidden states, pooling and CPU accumulation; excludes tokenization, model loading and saving',
                'source_sha256': {Path(__file__).name: file_hash(__file__),
                                  'tokenize_imdb.py': file_hash(base / 'tokenize_imdb.py')},
                'splits': {}}
    if device.type == 'cuda':
        metadata['gpu'] = torch.cuda.get_device_name(device)
    for split, rows in manifest['splits'].items():
        print(f'Extracting {source.name}: {split}', flush=True)
        features, labels, review_ids, counts, statistics = extract_split(
            model, rows, data_dir, word_to_id, args, device)
        metadata['splits'][split] = statistics
        np.savez(output / f'{split}.npz', features=features, labels=labels,
                 review_ids=review_ids, token_counts=counts)
        save_json(output / 'metadata.json', metadata)
        print(f'{split}: mean_max={features.shape}, '
              f'{statistics["extraction_seconds"]:.1f}s', flush=True)
    print(f'Features saved to {output}')


if __name__ == '__main__':
    main()
