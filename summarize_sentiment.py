# summarize sentiment classifiers for four LMs
import argparse
import csv
import json
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=base,
                        help='Parent directory containing results_lstm and the Transformer result directories.')
    parser.add_argument('--classifier-subdir', default='sentiment_classifier')
    parser.add_argument('--representations', nargs='+', default=[
        'results_lstm', 'results_transformer', 'results_transformer_skipgram',
        'results_transformer_glove'])
    args = parser.parse_args()
    rows = []
    common_setup = None
    model_contexts = {}
    checkpoints = {}
    details = []
    for representation in args.representations:
        folder = args.results_root / representation / args.classifier_subdir
        results = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        if results.get('protocol') != 'fixed_hyperparameters_single_seed':
            raise ValueError(f'Expected fixed-hyperparameter single-seed results: {folder}')
        metadata = results['features_metadata']
        if metadata.get('feature_schema_version') != 2 or metadata['pooling'] != 'mean_max':
            raise ValueError(f'Expected context-aware mean+max features: {folder}')
        checkpoints[representation] = metadata['checkpoint_sha256']
        shared = {}
        for key in ['manifest_sha256', 'vocab_sha256', 'pooling', 'pooling_exclusions',
                    'feature_dimension', 'chunk_length', 'truncation', 'language_model_frozen']:
            shared[key] = metadata[key]
        for key in ['seed', 'hidden_size', 'learning_rate', 'dropout',
                    'weight_decay', 'epochs', 'patience', 'batch_size', 'logistic_c', 'device']:
            shared[key] = config[key]
        if common_setup is None:
            common_setup = shared
        elif shared != common_setup:
            raise ValueError(f'Data or classifier comparison settings differ: {folder}')
        context = {'policy': metadata['context_policy'], 'stride': metadata['stride'],
                   'chunk_length': metadata['chunk_length']}
        model_type = metadata['model']
        if model_type not in model_contexts:
            model_contexts[model_type] = context
        elif context != model_contexts[model_type]:
            raise ValueError(f'Context policies differ for {model_type}: {folder}')
        lm_metrics_path = Path(metadata['checkpoint']).parent / 'metrics.json'
        lm_ppl = None
        if lm_metrics_path.exists():
            lm_metrics = json.loads(lm_metrics_path.read_text(encoding='utf-8'))
            lm_ppl = lm_metrics['test']['perplexity']
        logistic = results['logistic']
        mlp = results['mlp']
        for classifier in ['logistic', 'mlp']:
            if classifier == 'logistic':
                accuracy = logistic['test']['accuracy']
                macro_f1 = logistic['test']['macro_f1']
                parameters = ''
            else:
                accuracy = mlp['test']['accuracy']
                macro_f1 = mlp['test']['macro_f1']
                parameters = mlp['parameter_count']
            rows.append({'representation': representation, 'pooling': 'mean_max',
                         'classifier': classifier, 'feature_dimension': metadata['feature_dimension'],
                         'mlp_parameter_count': parameters, 'lm_test_ppl': lm_ppl,
                         'test_accuracy': accuracy, 'test_macro_f1': macro_f1,
                         'seed': results['seed']})
        details.append({'representation': representation, 'context': context,
                        'logistic_C': logistic['C'], 'mlp_settings': mlp['settings'],
                        'feature_extraction': metadata['splits'],
                        'classifier_timing': results['timing']})
    output = args.results_root / 'sentiment_comparison.csv'
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'shared_setup': common_setup,
               'checkpoint_sha256_by_representation': checkpoints,
               'context_by_model_type': model_contexts, 'results': rows, 'details': details,
               'notes': ['All representations concatenate mean and max pooling.',
                         'Compare representations within the same classifier family first.',
                         'LSTM carries recurrent state; Transformer uses finite overlapping windows. Architecture comparisons also reflect this context difference.',
                         'Lower LM perplexity is better; higher task accuracy and macro-F1 are better.',
                         'One fixed classifier seed is used; no seed-variability estimate is available.',
                         'Read error examples before interpreting failures.']}
    (args.results_root / 'sentiment_comparison.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for row in rows:
        print(f'{row["representation"]:30s} {row["classifier"]:9s} '
              f'accuracy={row["test_accuracy"]:.4f} macro-F1={row["test_macro_f1"]:.4f}')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
