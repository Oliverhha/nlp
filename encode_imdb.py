#!/usr/bin/env python3
# encode tokenized IMDb reviews with an existing vocabulary

import argparse
import hashlib
from pathlib import Path

SPLITS = ('train', 'validation', 'test')
SPECIALS = ('<pad>', '<unk>', '<bos>', '<eos>')


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=base / 'lm_tokenized')
    parser.add_argument('--vocab', type=Path, default=base / 'lm_vocab' / 'vocab.txt')
    parser.add_argument('--output-dir', type=Path, default=base / 'lm_encoded')
    args = parser.parse_args()
    source = args.input_dir.resolve()
    vocab_path = args.vocab.resolve()
    output = args.output_dir.resolve()
    if not vocab_path.is_file():
        parser.error(f'Vocabulary not found: {vocab_path}')
    if output == source or source in output.parents or output in source.parents:
        parser.error('Input and output directories must be separate, not nested.')
    if output == vocab_path or output in vocab_path.parents:
        parser.error('Output must not contain the source vocabulary.')
    if output.exists():
        parser.error(f'Output already exists: {output}. Choose a fresh --output-dir.')
    vocabulary = vocab_path.read_text(encoding='utf-8').splitlines()
    if any(not token or any(c.isspace() for c in token) for token in vocabulary):
        parser.error('Each vocabulary line must contain one nonempty token without whitespace.')
    if len(vocabulary) != len(set(vocabulary)):
        parser.error('Vocabulary contains duplicate tokens.')
    if tuple(vocabulary[:4]) != SPECIALS:
        parser.error('Expected special IDs: <pad>=0, <unk>=1, <bos>=2, <eos>=3.')
    word_to_id = {word: index for index, word in enumerate(vocabulary)}
    files_by_split = {}
    for split in SPLITS:
        folder = source / split
        if not folder.is_dir():
            parser.error(f'Missing input folder: {folder}')
        files_by_split[split] = sorted(folder.glob('*.txt'))
        if not files_by_split[split]:
            parser.error(f'No TXT reviews in: {folder}')
    output.mkdir(parents=True)
    # snapshot the exact ID mapping with the encoded data
    (output / 'vocab.txt').write_bytes(vocab_path.read_bytes())
    report = [f'Input: {source}', f'Vocabulary: {vocab_path}',
              f'Vocabulary size: {len(vocabulary)}',
              f'Vocabulary SHA256: {hashlib.sha256(vocab_path.read_bytes()).hexdigest()}',
              'Special IDs: PAD=0, UNK=1, BOS=2, EOS=3',
              'One TXT per review; first ID is BOS, last ID is EOS.',
              'UNK rate is computed on original text tokens, excluding BOS/EOS.',
              'No padding, truncation, chunking, or changes to vocabulary.']
    reserved = set(SPECIALS)
    for split, files in files_by_split.items():
        destination = output / split
        destination.mkdir()
        total = unknown = 0
        for path in files:
            tokens = path.read_text(encoding='utf-8').split()
            if not tokens:
                raise ValueError(f'Empty tokenized review: {path}. Output is incomplete.')
            if reserved.intersection(tokens):
                raise ValueError(f'Reserved token in text: {path}. Output is incomplete.')
            text_ids = [word_to_id.get(token, 1) for token in tokens]
            total += len(text_ids)
            unknown += text_ids.count(1)
            ids = [2] + text_ids + [3]
            (destination / path.name).write_text(' '.join(map(str, ids)) + '\n', encoding='utf-8')
        line = (f'{split}: documents={len(files)}, text_tokens={total}, '
                f'unknown_tokens={unknown}, UNK_rate={unknown / total:.4%}, '
                f'IDs_including_BOS_EOS={total + 2 * len(files)}')
        report.append(line)
        print(line)
    (output / 'encoding_report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'Saved to: {output}.')


if __name__ == '__main__':
    main()
