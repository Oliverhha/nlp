# prepare frozen 100d GloVe embeddings
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from lm_data import PAD, UNK, BOS, EOS, load_vocab


SOURCE_URL = 'https://nlp.stanford.edu/projects/glove/'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def coverage_by_split(data_dir, covered):
    results = {}
    for split in ['train', 'validation', 'test']:
        total = matched = 0
        paths = sorted((data_dir / split).glob('*.txt'))
        for path in paths:
            for item in path.read_text(encoding='utf-8').split():
                token_id = int(item)
                if token_id in (PAD, BOS, EOS):
                    continue
                total += 1
                if covered[token_id]:
                    matched += 1
        results[split] = {
            'documents': len(paths), 'text_tokens': total,
            'tokens_with_glove_vectors': matched,
            'token_coverage_percent': 100 * matched / total if total else None,
            'note': 'PAD/BOS/EOS excluded; UNK counts as uncovered.',
        }
    return results


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glove-file', type=Path, default=base / 'glove' / 'glove.6B.100d.txt')
    parser.add_argument('--data-dir', type=Path, default=base / 'lm_encoded')
    parser.add_argument('--output-dir', type=Path, default=base / 'results_glove')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--missing-std', type=float, default=0.1)
    args = parser.parse_args()
    if not args.glove_file.is_file():
        parser.error('GloVe file missing. Download and extract glove.6B.100d.txt first.')
    if args.output_dir.exists():
        parser.error('Output exists; choose a fresh --output-dir.')
    if args.missing_std <= 0:
        parser.error('missing-std must be positive.')
    vocab = load_vocab(args.data_dir)
    dimension = 100
    word_to_id = {word: index for index, word in enumerate(vocab) if index >= 4}

    # use the same random fallback policy as Skip-gram
    rng = np.random.default_rng(args.seed)
    matrix = rng.normal(0, args.missing_std,
                        size=(len(vocab), dimension)).astype(np.float32)
    matrix[PAD] = 0
    covered = [False] * len(vocab)
    source_hash = hashlib.sha256()
    rows_read = 0
    print('Reading GloVe and matching the existing vocabulary...', flush=True)
    start = time.perf_counter()
    # stream the file, retain only vectors required by the fixed LM vocabulary
    with args.glove_file.open('rb') as stream:
        for raw_line in stream:
            source_hash.update(raw_line)
            parts = raw_line.decode('utf-8').split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError('Expected a word followed by its 100 GloVe coordinates.')
            word, coordinates = parts
            rows_read += 1
            if rows_read == 1 and len(coordinates.split()) != dimension:
                raise ValueError('This experiment requires the 100d GloVe file.')
            if word not in word_to_id:
                continue
            vector = np.asarray(coordinates.split(), dtype=np.float32)
            if vector.shape != (dimension,) or not np.isfinite(vector).all():
                raise ValueError(f'Invalid GloVe vector for {word!r}.')
            token_id = word_to_id[word]
            matrix[token_id] = vector
            covered[token_id] = True
    loading_seconds = time.perf_counter() - start

    # keep unmatched words in the vocabulary with their seeded random vectors
    missing = [word for index, word in enumerate(vocab) if index >= 4 and not covered[index]]
    matched = sum(covered)
    start = time.perf_counter()
    coverage = {
        'lexical_vocabulary_size': len(vocab) - 4,
        'matched_lexical_tokens': matched,
        'missing_lexical_tokens': len(missing),
        'vocabulary_coverage_percent': 100 * matched / (len(vocab) - 4),
        'splits': coverage_by_split(args.data_dir, covered),
    }
    coverage_seconds = time.perf_counter() - start
    initialization = {
        'missing_words': f'Normal(mean=0, std={args.missing_std}); seed={args.seed}',
        'special_tokens': 'PAD is zero; UNK/BOS/EOS use the same seeded random initialization.',
        'freezing': 'The entire aligned input table will be frozen in the language model.',
        'normalization': 'Public vectors are copied without scaling or normalization.',
    }
    vocab_hash = hashlib.sha256((args.data_dir / 'vocab.txt').read_bytes()).hexdigest()
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(embedding_dim=dimension, method='GloVe 6B 100d',
                  source_url=SOURCE_URL, source_file_sha256=source_hash.hexdigest(),
                  external_corpus='Wikipedia 2014 + Gigaword 5, 6 billion tokens, uncased')

    # use the same checkpoint format as Skip-gram for the frozen trainer
    args.output_dir.mkdir(parents=True)
    start = time.perf_counter()
    torch.save({'embedding_matrix': torch.from_numpy(matrix),
                'vocabulary': vocab, 'vocab_sha256': vocab_hash,
                'embedding_dim': dimension, 'method': 'GloVe 6B 100d',
                'coverage': coverage, 'initialization': initialization,
                'config': config}, args.output_dir / 'embedding.pt')
    (args.output_dir / 'missing_words.txt').write_text('\n'.join(missing), encoding='utf-8')
    save_seconds = time.perf_counter() - start
    save_json(args.output_dir / 'config.json', config)
    save_json(args.output_dir / 'metrics.json', {
        'method': 'GloVe 6B 100d', 'config': config,
        'source_rows_read': rows_read, 'coverage': coverage,
        'initialization': initialization, 'vocab_sha256': vocab_hash,
        'timing': {'file_loading_and_alignment_seconds': loading_seconds,
                   'coverage_measurement_seconds': coverage_seconds,
                   'matrix_save_seconds': save_seconds,
                   'note': 'Preparation costs only; no local GloVe training. Download/extraction time excluded.'},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    })
    print(f'Saved {args.output_dir / "embedding.pt"}', flush=True)
    print(f'Vocabulary coverage: {coverage["vocabulary_coverage_percent"]:.2f}%')
    print(f'Missing lexical tokens: {len(missing)}')


if __name__ == '__main__':
    main()
