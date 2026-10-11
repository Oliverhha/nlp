# train 100d Skip-gram on the language-model train corpus only.
import argparse
import hashlib
import json
import platform
import sys
import time
import zlib
from pathlib import Path

import gensim
import numpy as np
import torch
from gensim.models import Word2Vec
from lm_data import PAD, UNK, BOS, EOS, load_vocab


class ReviewCorpus:
    def __init__(self, folder):
        self.paths = sorted(folder.glob('*.txt'))

    def __iter__(self):
        for path in self.paths:
            yield path.read_text(encoding='utf-8').split()


def stable_hash(word):
    # avoid Python's process-dependent string hash during initialization.
    return zlib.crc32(word.encode('utf-8'))


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
            'tokens_with_skipgram_vectors': matched,
            'token_coverage_percent': 100 * matched / total if total else None,
            'note': 'PAD/BOS/EOS excluded; UNK counts as uncovered.',
        }
    return results


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, default=base / 'lm_tokenized/train')
    parser.add_argument('--data-dir', type=Path, default=base / 'lm_encoded')
    parser.add_argument('--output-dir', type=Path, default=base / 'results_skipgram')
    parser.add_argument('--embedding-dim', type=int, default=100)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--negative', type=int, default=5)
    parser.add_argument('--min-count', type=int, default=2)
    parser.add_argument('--sample', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--missing-std', type=float, default=0.1)
    args = parser.parse_args()
    if min(args.embedding_dim, args.window, args.negative, args.min_count,
           args.epochs, args.workers) < 1 or args.sample < 0 or args.missing_std <= 0:
        parser.error('Invalid Skip-gram hyperparameter range.')
    if args.output_dir.exists():
        parser.error('Output exists; choose a fresh --output-dir.')
    vocab = load_vocab(args.data_dir)
    corpus = ReviewCorpus(args.train_dir)
    if not corpus.paths:
        parser.error('No training reviews found.')
    encoded_names = {p.name for p in (args.data_dir / 'train').glob('*.txt')}
    if {p.name for p in corpus.paths} != encoded_names:
        parser.error('Tokenized and encoded training review filenames differ.')
    args.output_dir.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(method='Skip-gram with negative sampling', sg=1, hs=0,
                  alpha=0.025, min_alpha=0.0001, shrink_windows=True)
    save_json(args.output_dir / 'config.json', config)

    model = Word2Vec(vector_size=args.embedding_dim, window=args.window,
                     min_count=args.min_count, sg=1, negative=args.negative,
                     hs=0, sample=args.sample, workers=args.workers,
                     seed=args.seed, alpha=0.025, min_alpha=0.0001,
                     hashfxn=stable_hash, shrink_windows=True)
    print('Building Skip-gram vocabulary from training reviews...', flush=True)
    start = time.perf_counter()
    model.build_vocab(corpus)
    build_seconds = time.perf_counter() - start

    print(f'Training Skip-gram for {args.epochs} epochs on CPU...', flush=True)
    start = time.perf_counter()
    effective_words, raw_words = model.train(
        corpus, total_examples=model.corpus_count, epochs=args.epochs)
    train_seconds = time.perf_counter() - start

    rng = np.random.default_rng(args.seed)
    matrix = rng.normal(0, args.missing_std,
                        size=(len(vocab), args.embedding_dim)).astype(np.float32)
    covered = [False] * len(vocab)
    missing = []
    for token_id, word in enumerate(vocab):
        if token_id in (PAD, UNK, BOS, EOS):
            continue
        if word in model.wv:
            matrix[token_id] = model.wv[word]
            covered[token_id] = True
        else:
            missing.append(word)
    matrix[PAD] = 0
    vocab_hash = hashlib.sha256((args.data_dir / 'vocab.txt').read_bytes()).hexdigest()
    matched = sum(covered)
    coverage = {
        'lexical_vocabulary_size': len(vocab) - 4,
        'matched_lexical_tokens': matched,
        'missing_lexical_tokens': len(missing),
        'vocabulary_coverage_percent': 100 * matched / (len(vocab) - 4),
        'splits': coverage_by_split(args.data_dir, covered),
    }
    initialization = {
        'missing_words': f'Normal(mean=0, std={args.missing_std}); seed={args.seed}',
        'special_tokens': 'PAD is zero; UNK/BOS/EOS use the same seeded random initialization.',
        'freezing': 'The entire aligned input table will be frozen in the language model.',
        'normalization': 'Learned vectors are copied without scaling or normalization.',
    }

    # Step 4: save the vectors and the matrix used by the Transformer.
    start = time.perf_counter()
    model.wv.save(str(args.output_dir / 'skipgram_vectors.kv'))
    torch.save({'embedding_matrix': torch.from_numpy(matrix),
                'vocabulary': vocab, 'vocab_sha256': vocab_hash,
                'embedding_dim': args.embedding_dim, 'method': 'Skip-gram',
                'coverage': coverage, 'initialization': initialization,
                'config': config}, args.output_dir / 'embedding.pt')
    save_seconds = time.perf_counter() - start
    (args.output_dir / 'missing_words.txt').write_text('\n'.join(missing), encoding='utf-8')
    metrics = {
        'method': 'Skip-gram', 'config': config,
        'training_documents': model.corpus_count,
        'training_raw_tokens_per_pass': model.corpus_total_words,
        'skipgram_vocabulary_size': len(model.wv),
        'effective_training_words_all_epochs': effective_words,
        'raw_training_words_all_epochs': raw_words,
        'timing': {'build_vocabulary_seconds': build_seconds,
                   'vector_training_seconds': train_seconds,
                   'embedding_fit_seconds': build_seconds + train_seconds,
                   'vector_and_matrix_save_seconds': save_seconds},
        'coverage': coverage, 'initialization': initialization,
        'vocab_sha256': vocab_hash,
        'hardware': {'device': 'cpu', 'platform': platform.platform(),
                     'python_version': sys.version, 'workers': args.workers,
                     'gensim_version': gensim.__version__,
                     'numpy_version': np.__version__},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    save_json(args.output_dir / 'metrics.json', metrics)
    print(f'Saved {args.output_dir / "embedding.pt"}', flush=True)
    print(f'Vocabulary coverage: {coverage["vocabulary_coverage_percent"]:.2f}%')
    print(f'Embedding fit time: {build_seconds + train_seconds:.2f}s')


if __name__ == '__main__':
    main()
