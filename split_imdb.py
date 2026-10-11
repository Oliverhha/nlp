#!/usr/bin/env python3
"""split cleaned, unlabeled IMDb files into train/validation/test.
Defaults: 25000 / 5000 / 5000 reviews, seed 42.
"""
import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=base / 'aclImdb_cleaned')
    parser.add_argument('--output-dir', type=Path, default=base / 'lm_splits')
    parser.add_argument('--train-size', type=int, default=25000)
    parser.add_argument('--val-size', type=int, default=5000)
    parser.add_argument('--test-size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    source = data_dir / 'train' / 'unsup'
    output = args.output_dir.resolve()
    if not source.is_dir():
        parser.error(f'Unlabeled directory not found: {source}')
    if output == data_dir or data_dir in output.parents or output in data_dir.parents:
        parser.error('Output and dataset directories must be separate, not nested.')
    if output.exists():
        parser.error(f'Output already exists: {output}. Choose a fresh --output-dir.')
    sizes = (args.train_size, args.val_size, args.test_size)
    if any(size <= 0 for size in sizes):
        parser.error('All split sizes must be positive.')
    files = sorted(source.glob('*.txt'))
    if sum(sizes) > len(files):
        parser.error(f'Requested {sum(sizes)} reviews, but found only {len(files)}.')
    random.Random(args.seed).shuffle(files)
    train_end = args.train_size
    val_end = train_end + args.val_size
    test_end = val_end + args.test_size
    splits = {'train': files[:train_end],
              'validation': files[train_end:val_end],
              'test': files[val_end:test_end]}
    output.mkdir(parents=True)
    for name, paths in splits.items():
        target = output / name
        target.mkdir()
        for path in paths:
            shutil.copyfile(path, target / path.name)
        print(f'{name}: {len(paths)} TXT files -> {target}')
    print(f'Seed: {args.seed}; unused: {len(files) - test_end} reviews.')

if __name__ == '__main__':
    main()
