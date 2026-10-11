#!/usr/bin/env python3
# tokenize split IMDb
import argparse
import re
from pathlib import Path

C1_PATTERN = re.compile(r"[\u0080-\u009f]")
WORD_APOSTROPHE_PATTERN = re.compile(
    r"(?<=[^\W\d_])[‘’ʼ＇`´](?=[^\W\d_])", flags=re.UNICODE
)
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", flags=re.UNICODE)
SPLITS = ('train', 'validation', 'test')


def normalize_for_tokenization(text: str) -> str:
    """Conservative character fixes; preserve whitespace boundaries and spelling."""
    text = C1_PATTERN.sub(' ', text)
    return WORD_APOSTROPHE_PATTERN.sub("'", text)


def tokenize(text: str) -> list[str]:
    """Use the same function later for prompts and downstream labeled reviews."""
    return TOKEN_PATTERN.findall(normalize_for_tokenization(text))


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=base / 'lm_splits')
    parser.add_argument('--output-dir', type=Path, default=base / 'lm_tokenized')
    args = parser.parse_args()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        parser.error('Input and output directories must be separate, not nested.')
    if output.exists():
        parser.error(f'Output already exists: {output}. Choose a fresh --output-dir.')
    files_by_split = {}
    for split in SPLITS:
        folder = source / split
        if not folder.is_dir():
            parser.error(f'Missing input folder: {folder}')
        files_by_split[split] = sorted(folder.glob('*.txt'))
        if not files_by_split[split]:
            parser.error(f'No TXT reviews found in: {folder}')
    output.mkdir(parents=True)
    for split, files in files_by_split.items():
        destination = output / split
        destination.mkdir()
        token_count = 0
        empty_count = 0
        controls_replaced = 0
        apostrophes_normalized = 0
        for path in files:
            text = path.read_text(encoding='utf-8')
            controls_replaced += len(C1_PATTERN.findall(text))
            apostrophes_normalized += len(WORD_APOSTROPHE_PATTERN.findall(C1_PATTERN.sub(' ', text)))
            tokens = tokenize(text)
            token_count += len(tokens)
            empty_count += int(not tokens)
            (destination / path.name).write_text(' '.join(tokens) + '\n', encoding='utf-8')
        print(f'{split}: {len(files)} documents, {token_count} tokens, '
              f'{empty_count} empty documents -> {destination}')
        print(f'  C1 controls replaced: {controls_replaced}; '
              f'word-internal apostrophe variants normalized: {apostrophes_normalized}')


if __name__ == '__main__':
    main()
