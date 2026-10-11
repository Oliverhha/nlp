#!/usr/bin/env python3
# build vocabulary from lm_tokenized/train only and report split coverage
import argparse
from collections import Counter
from pathlib import Path

SPECIALS = ('<pad>', '<unk>', '<bos>', '<eos>')
SPLITS = ('train', 'validation', 'test')


def read_tokens(path: Path) -> list[str]:
    return path.read_text(encoding='utf-8').split()


def length_summary(lengths: list[int]) -> str:
    ordered = sorted(lengths)
    def percentile(p: float) -> int:
        return ordered[max(0, min(len(ordered) - 1, int((len(ordered) - 1) * p)))]
    return (f'mean={sum(lengths)/len(lengths):.2f}, '
            f'median_lower={percentile(0.5)}, p95_lower={percentile(0.95)}, '
            f'max={ordered[-1]}')


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=base / 'lm_tokenized')
    parser.add_argument('--output-dir', type=Path, default=base / 'lm_vocab')
    parser.add_argument('--max-vocab-size', type=int, default=15000,
                        help='Total vocabulary cap including four special tokens.')
    parser.add_argument('--min-freq', type=int, default=2)
    args = parser.parse_args()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if args.max_vocab_size <= len(SPECIALS) or args.min_freq < 1:
        parser.error('Vocabulary cap must exceed 4 and min-freq must be positive.')
    if output == source or source in output.parents or output in source.parents:
        parser.error('Input and output directories must be separate, not nested.')
    if output.exists():
        parser.error(f'Output already exists: {output}. Choose a fresh --output-dir.')
    paths = {}
    for split in SPLITS:
        folder = source / split
        if not folder.is_dir():
            parser.error(f'Missing folder: {folder}')
        paths[split] = sorted(folder.glob('*.txt'))
        if not paths[split]:
            parser.error(f'No TXT files in: {folder}')

    counts = Counter()
    for path in paths['train']:
        counts.update(read_tokens(path))
    collisions = set(SPECIALS) & counts.keys()
    if collisions:
        parser.error(f'Text contains reserved tokens: {sorted(collisions)}')
    if not counts:
        parser.error('Training corpus contains no tokens.')
    # only training frequencies select vocabulary
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    selected = [word for word, freq in ranked if freq >= args.min_freq]
    selected = selected[:args.max_vocab_size - len(SPECIALS)]
    vocabulary = list(SPECIALS) + selected
    known_words = set(selected)
    output.mkdir(parents=True)
    (output / 'vocab.txt').write_text('\n'.join(vocabulary) + '\n', encoding='utf-8')
    with (output / 'train_word_counts.tsv').open('w', encoding='utf-8') as handle:
        handle.write('token\ttrain_frequency\n')
        for word, freq in ranked:
            handle.write(f'{word}\t{freq}\n')
    report = [f'Vocabulary source: {source / "train"}',
              f'Total vocabulary size: {len(vocabulary)}',
              f'Vocabulary cap (including specials): {args.max_vocab_size}',
              f'Minimum training frequency: {args.min_freq}',
              f'Training distinct tokens: {len(counts)}',
              'Special IDs: <pad>=0, <unk>=1, <bos>=2, <eos>=3',
              'vocab.txt: zero-based line index is token ID.',
              'Coverage excludes special tokens; no BOS/EOS have been added.',
              'Validation/test statistics do not affect vocabulary selection.']
    for split in SPLITS:
        total = unknown = empty = 0
        lengths = []
        for path in paths[split]:
            tokens = read_tokens(path)
            lengths.append(len(tokens))
            total += len(tokens)
            unknown += sum(token not in known_words for token in tokens)
            empty += int(not tokens)
        rate = unknown / total if total else 0.0
        line = (f'{split}: documents={len(paths[split])}, tokens={total}, '
                f'unknown_tokens={unknown}, UNK_rate={rate:.4%}, '
                f'empty_documents={empty}; {length_summary(lengths)}')
        report.append(line)
        print(line)
    (output / 'vocabulary_report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'Vocabulary: {len(vocabulary)} tokens -> {output / "vocab.txt"}')


if __name__ == '__main__':
    main()
