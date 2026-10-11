# create labeled IMDb split as train/validation/test
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import sklearn
from sklearn.model_selection import train_test_split


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def collect_reviews(source, official_split):
    rows = []
    for category, label in [('neg', 0), ('pos', 1)]:
        folder = source / official_split / category
        files = sorted(folder.glob('*.txt'))
        if len(files) != 12500:
            raise ValueError(f'Expected 12500 labeled reviews in {folder}.')
        for path in files:
            rows.append({'source_review_id': path.relative_to(source).as_posix(),
                         'category': category, 'label': label, 'sha256': file_hash(path)})
    return rows


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=base / 'aclImdb_cleaned')
    parser.add_argument('--output-dir', type=Path, default=base / 'sentiment_splits')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        parser.error('Input and output directories must be separate, not nested.')
    if output.exists():
        parser.error(f'Output already exists: {output}. Choose a fresh --output-dir.')

    official_train = collect_reviews(source, 'train')
    official_test = collect_reviews(source, 'test')
    labels = [row['label'] for row in official_train]
    train_rows, validation_rows = train_test_split(
        official_train, test_size=5000, random_state=args.seed, stratify=labels)
    manifest = {'dataset_format': 'imdb_sentiment_txt',
                'source_dir': str(source), 'data_dir': str(output),
                'split_seed': args.seed, 'sklearn_version': sklearn.__version__,
                'source_sha256': file_hash(__file__),
                'label_names': {'0': 'negative', '1': 'positive'},
                'counts': {}, 'splits': {}}
    output.mkdir(parents=True)
    for split, rows in [('train', train_rows), ('validation', validation_rows),
                        ('test', official_test)]:
        for category in ['neg', 'pos']:
            (output / split / category).mkdir(parents=True)
        saved_rows = []
        for row in sorted(rows, key=lambda item: item['source_review_id']):
            original = source / row['source_review_id']
            destination = output / split / row['category'] / original.name
            shutil.copy2(original, destination)
            saved_rows.append({'review_id': destination.relative_to(output).as_posix(),
                               'source_review_id': row['source_review_id'],
                               'label': row['label'], 'sha256': row['sha256']})
        manifest['splits'][split] = saved_rows
        negatives = sum(row['label'] == 0 for row in saved_rows)
        positives = sum(row['label'] == 1 for row in saved_rows)
        manifest['counts'][split] = {'total': len(saved_rows),
                                     'negative': negatives, 'positive': positives}
        print(f'{split}: {len(saved_rows)} reviews, neg={negatives}, pos={positives}', flush=True)
    (output / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'dataset saved to {output}')


if __name__ == '__main__':
    main()
