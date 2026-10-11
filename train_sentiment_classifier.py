# train MLP and logistic regression on a saved set of frozen LM features.
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import time
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def save_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    return {'accuracy': float(accuracy_score(labels, predictions)),
            'macro_f1': float(f1_score(labels, predictions, average='macro', zero_division=0)),
            'classification_report': classification_report(
                labels, predictions, labels=[0, 1], target_names=['negative', 'positive'],
                output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}


class SentimentMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout=0.2):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, features):
        hidden = self.activation(self.hidden(features))
        return self.output(self.dropout(hidden))


@torch.inference_mode()
def predict_mlp(model, features, batch_size, device):
    model.eval()
    probabilities = []
    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size].to(device)
        probabilities.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def train_mlp(train_x, train_y, validation_x, validation_y, settings, seed, args, device):
    set_seed(seed)
    model = SentimentMLP(train_x.shape[1], args.hidden_size, settings['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=args.weight_decay)
    loss_function = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_ce = float('inf')
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    if device.type == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            features = train_x[indices].to(device)
            labels = train_y[indices].to(device)
            optimizer.zero_grad()
            loss = loss_function(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        probabilities = predict_mlp(model, validation_x, args.batch_size, device)
        validation_f1 = float(f1_score(validation_y, probabilities.argmax(axis=1),
                                       average='macro', zero_division=0))
        correct_probabilities = probabilities[np.arange(len(validation_y)), validation_y]
        validation_ce = float(-np.log(np.maximum(correct_probabilities, 1e-12)).mean())
        history.append({'epoch': epoch, 'train_ce': total_loss / len(train_x),
                        'validation_ce': validation_ce, 'validation_macro_f1': validation_f1})
        # Select by validation macro-F1; use CE to break exact ties.
        improved = validation_f1 > best_f1 or (validation_f1 == best_f1 and validation_ce < best_ce)
        if improved:
            best_f1 = validation_f1
            best_ce = validation_ce
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f'  seed={seed} epoch={epoch}: train CE={history[-1]["train_ce"]:.4f}, '
                  f'val F1={validation_f1:.4f}, best={best_epoch}', flush=True)
        if stale_epochs >= args.patience:
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    information = {'settings': settings, 'seed': seed, 'best_epoch': best_epoch,
                   'actual_epochs': len(history), 'best_validation_macro_f1': best_f1,
                   'best_validation_ce': best_ce, 'fit_seconds_including_validation': seconds,
                   'stop_reason': 'early_stopping' if stale_epochs >= args.patience else 'epoch_limit'}
    return model, history, information


def save_curve(path, history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row['epoch'] for row in history]
    axes[0].plot(epochs, [row['train_ce'] for row in history], label='Train (dropout active)')
    axes[0].plot(epochs, [row['validation_ce'] for row in history], label='Validation')
    axes[0].set_ylabel('Cross-entropy')
    axes[0].legend()
    axes[1].plot(epochs, [row['validation_macro_f1'] for row in history])
    axes[1].set_ylabel('Validation macro-F1')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_predictions(folder, labels, probabilities, review_ids, data_dir):
    rows = []
    for index, review_id in enumerate(review_ids):
        predicted = int(probabilities[index].argmax())
        rows.append({'review_id': str(review_id), 'true_label': int(labels[index]),
                     'predicted_label': predicted, 'positive_probability': float(probabilities[index, 1]),
                     'predicted_confidence': float(probabilities[index, predicted])})
    save_csv(folder / 'test_predictions.csv', rows)
    examples = {}
    for label, name in [(0, 'negative_predicted_positive'), (1, 'positive_predicted_negative')]:
        wrong = [row for row in rows if row['true_label'] == label and row['predicted_label'] != label]
        wrong.sort(key=lambda row: row['predicted_confidence'], reverse=True)
        examples[name] = wrong[:10]
    uncertain = sorted(rows, key=lambda row: abs(row['positive_probability'] - 0.5))
    examples['near_decision_boundary'] = [dict(row) for row in uncertain[:10]]
    for group in examples.values():
        for row in group:
            path = data_dir / row['review_id']
            row['review'] = path.read_text(encoding='utf-8') if path.exists() else None
    save_json(folder / 'error_examples.json', {
        'selection': '10 highest-confidence errors per true class and 10 least-confident test predictions; inspect manually, not a random sample',
        'label_names': {'0': 'negative', '1': 'positive'}, 'examples': examples})


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--logistic-c', type=float, default=1.0)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.hidden_size, args.cpu_threads) < 1:
        parser.error('Epochs, patience, sizes and threads must be positive.')
    if args.learning_rate <= 0 or args.logistic_c <= 0:
        parser.error('Learning rate and logistic C must be positive.')
    if args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error('Weight decay must be nonnegative and dropout must lie in [0, 1).')
    source = args.features_dir.resolve()
    metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('feature_schema_version') != 2:
        parser.error('Use context-aware mean+max features from the model feature directory.')
    if metadata.get('pooling') != 'mean_max':
        parser.error('Only concatenated mean+max features are supported.')
    model_results = Path(metadata['checkpoint']).parent
    output = args.output_dir or model_results / 'sentiment_classifier'
    if output.exists():
        parser.error(f'Output already exists: {output}. Choose a fresh --output-dir.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu or your training environment.')
    torch.set_num_threads(args.cpu_threads)
    threadpool_limits(limits=args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    warnings.filterwarnings('error', category=ConvergenceWarning)
    output.mkdir(parents=True)
    started_all = time.perf_counter()
    data = {}
    for split in ['train', 'validation', 'test']:
        with np.load(source / f'{split}.npz', allow_pickle=False) as stored:
            data[split] = {key: stored[key] for key in stored.files}
    scaler = StandardScaler()
    train_x = scaler.fit_transform(data['train']['features']).astype(np.float32)
    validation_x = scaler.transform(data['validation']['features']).astype(np.float32)
    # validation selects the MLP checkpoint only; hyperparameters are fixed.
    train_y = data['train']['labels']
    validation_y = data['validation']['labels']
    joblib.dump(scaler, output / 'scaler.joblib')
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    config['protocol'] = 'fixed_hyperparameters_single_seed'
    config['selection'] = 'fixed hyperparameters; best MLP epoch by validation macro-F1, ties by validation CE'
    config['input_dimension'] = metadata['feature_dimension']
    config['features_metadata'] = metadata
    config['hardware'] = {'platform': platform.platform(), 'cpu': platform.processor(),
                          'cpu_threads': args.cpu_threads, 'mlp_device': str(device),
                          'logistic_device': 'cpu'}
    if device.type == 'cuda':
        config['hardware']['gpu'] = torch.cuda.get_device_name(device)
    config['versions'] = {'torch': str(torch.__version__), 'numpy': np.__version__,
                          'sklearn': sklearn.__version__}
    config['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save_json(output / 'config.json', config)

    logistic_started = time.perf_counter()
    classifier = LogisticRegression(C=args.logistic_c, solver='lbfgs', max_iter=2000,
                                    random_state=args.seed)
    classifier.fit(train_x, train_y)
    logistic_fit_seconds = time.perf_counter() - logistic_started

    train_tensor = torch.from_numpy(train_x)
    train_label_tensor = torch.from_numpy(train_y)
    validation_tensor = torch.from_numpy(validation_x)
    settings = {'learning_rate': args.learning_rate, 'dropout': args.dropout}
    print(f'Training MLP once: {settings}, seed={args.seed}', flush=True)
    model, history, information = train_mlp(
        train_tensor, train_label_tensor, validation_tensor, validation_y,
        settings, args.seed, args, device)

    # evaluate the fixed logistic model and the best validation MLP checkpoint.
    test_x = scaler.transform(data['test']['features']).astype(np.float32)
    test_y = data['test']['labels']
    test_ids = data['test']['review_ids']
    data_dir = Path(metadata['data_dir'])
    logistic_folder = output / 'logistic'
    logistic_folder.mkdir()
    classifier.predict_proba(test_x[:args.batch_size])
    started = time.perf_counter()
    probabilities = classifier.predict_proba(test_x)
    logistic_test_seconds = time.perf_counter() - started
    logistic_results = {'C': args.logistic_c, 'seed': args.seed,
                        'validation': task_metrics(validation_y, classifier.predict_proba(validation_x)),
                        'test': task_metrics(test_y, probabilities),
                        'fit_seconds': logistic_fit_seconds,
                        'test_classifier_seconds': logistic_test_seconds,
                        'test_documents_per_second': len(test_y) / logistic_test_seconds}
    save_predictions(logistic_folder, test_y, probabilities, test_ids, data_dir)
    joblib.dump({'classifier': classifier, 'scaler': scaler}, logistic_folder / 'classifier.joblib')
    logistic_results['saved_model_bytes'] = (logistic_folder / 'classifier.joblib').stat().st_size
    save_json(logistic_folder / 'metrics.json', logistic_results)

    test_tensor = torch.from_numpy(test_x)
    folder = output / 'mlp'
    folder.mkdir()
    validation_probabilities = predict_mlp(model, validation_tensor, args.batch_size, device)
    predict_mlp(model, test_tensor[:args.batch_size], args.batch_size, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    probabilities = predict_mlp(model, test_tensor, args.batch_size, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    test_seconds = time.perf_counter() - started
    information['validation'] = task_metrics(validation_y, validation_probabilities)
    information['test'] = task_metrics(test_y, probabilities)
    information['test_classifier_seconds'] = test_seconds
    information['test_documents_per_second'] = len(test_y) / test_seconds
    information['parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({'model_state': state, 'input_dim': train_x.shape[1],
                'hidden_size': args.hidden_size, 'settings': settings,
                'seed': args.seed, 'best_epoch': information['best_epoch'],
                'scaler_mean': torch.tensor(scaler.mean_),
                'scaler_scale': torch.tensor(scaler.scale_),
                'features_metadata': metadata}, folder / 'best_classifier.pt')
    information['saved_model_bytes'] = (folder / 'best_classifier.pt').stat().st_size
    save_json(folder / 'metrics.json', information)
    save_csv(folder / 'history.csv', history)
    save_curve(folder / 'loss_curves.png', history)
    save_predictions(folder, test_y, probabilities, test_ids, data_dir)
    summary = {'protocol': config['protocol'],
               'representation': metadata['representation'], 'features_metadata': metadata,
               'seed': args.seed, 'logistic': logistic_results, 'mlp': information,
               'timing': {'logistic_fit_seconds': logistic_fit_seconds,
                          'mlp_fit_seconds_including_validation': information['fit_seconds_including_validation'],
                          'whole_script_seconds': time.perf_counter() - started_all}}
    save_json(output / 'metrics.json', summary)
    print(f'Logistic: test accuracy={logistic_results["test"]["accuracy"]:.4f}, '
          f'macro-F1={logistic_results["test"]["macro_f1"]:.4f}')
    print(f'MLP seed={args.seed}: test accuracy={information["test"]["accuracy"]:.4f}, '
          f'macro-F1={information["test"]["macro_f1"]:.4f}')
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
