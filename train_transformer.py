# train transformer, select the best validation model and evaluate test.

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from lm_data import PAD, ChunkDataset, collate_chunks, load_vocab
from transformer_model import TransformerLanguageModel

def hardware_info():
    # read basic CPU and memory information on Linux.
    info = {
        'platform': platform.platform(),
        'cpu_model': platform.processor(),
        'logical_cpus': os.cpu_count(),
    }
    cpu_file = Path('/proc/cpuinfo')
    if cpu_file.exists():
        for line in cpu_file.read_text().splitlines():
            if line.startswith('model name'):
                info['cpu_model'] = line.split(':', 1)[1].strip()
                break
    memory_file = Path('/proc/meminfo')
    if memory_file.exists():
        for line in memory_file.read_text().splitlines():
            if line.startswith('MemTotal:'):
                info['system_ram_bytes'] = int(line.split()[1]) * 1024
    info['thread_environment'] = {}
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        info['thread_environment'][name] = os.environ.get(name)
    if hasattr(os, 'sched_getaffinity'):
        info['available_logical_cpus'] = len(os.sched_getaffinity(0))
    try:
        from threadpoolctl import threadpool_info
        info['numpy_threadpools'] = threadpool_info()
    except ImportError:
        info['numpy_threadpools'] = None
    return info

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def perplexity(ce):
    return math.exp(ce) if ce < 700 else float('inf')

def telemetry():
    try:
        p = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,temperature.gpu,power.draw,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None

def dataset_summary(dataset):
    return {'documents': len(dataset.docs), 'chunks': len(dataset), 'target_tokens': sum((len(ids) - 1 for ids in dataset.docs))}

def train_one_epoch(model, loader, device, optimizer, clip):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction='sum')
    total_loss = 0.0
    total_tokens = 0
    steps = 0
    sync(device)
    start = time.perf_counter()
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        valid_tokens = targets.ne(PAD).sum().item()
        optimizer.zero_grad()
        logits = model(inputs)
        logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        loss_sum = criterion(logits, flat_targets)
        if not torch.isfinite(loss_sum):
            raise RuntimeError('Nonfinite training loss.')
        # PAD is ignored, divide by real target count before backward
        loss = loss_sum / valid_tokens
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss_sum.item()
        total_tokens += valid_tokens
        steps += 1
    sync(device)
    seconds = time.perf_counter() - start
    average_loss = total_loss / total_tokens
    return {'cross_entropy': average_loss, 'perplexity': perplexity(average_loss), 'target_tokens': total_tokens, 'steps': steps, 'seconds': seconds, 'target_tokens_per_second': total_tokens / seconds}

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction='sum')
    total_loss = 0.0
    total_tokens = 0
    steps = 0
    sync(device)
    start = time.perf_counter()
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss_sum = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if not torch.isfinite(loss_sum):
            raise RuntimeError('Nonfinite evaluation loss.')
        total_loss += loss_sum.item()
        total_tokens += targets.ne(PAD).sum().item()
        steps += 1
    sync(device)
    seconds = time.perf_counter() - start
    average_loss = total_loss / total_tokens
    return {'cross_entropy': average_loss, 'perplexity': perplexity(average_loss), 'target_tokens': total_tokens, 'steps': steps, 'seconds': seconds, 'target_tokens_per_second': total_tokens / seconds}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def plot_history(history, path):
    os.environ.setdefault('MPLCONFIGDIR', str(path.parent / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        [r['epoch'] for r in history],
        [r['train_cross_entropy'] for r in history],
        label='Train (dropout active)',
        marker='o',
    )
    ax.plot(
        [r['epoch'] for r in history],
        [r['validation_cross_entropy'] for r in history],
        label='Validation (eval)',
        marker='o',
    )
    ax.set(xlabel='Epoch', ylabel='Token-average cross-entropy', title='Transformer language-model learning curves')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=base / 'lm_encoded')
    p.add_argument('--output-dir', type=Path, default=base / 'results_transformer')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--embedding-dim', type=int, default=100)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--num-heads', type=int, default=4)
    p.add_argument('--feedforward-dim', type=int, default=512)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--context-length', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)
    args = p.parse_args()
    positive = [args.embedding_dim, args.d_model, args.num_heads, args.feedforward_dim, args.num_layers, args.context_length, args.batch_size, args.epochs, args.patience, args.learning_rate, args.grad_clip, args.cpu_threads]
    if min(positive) <= 0 or not 0 <= args.dropout < 1 or args.weight_decay < 0:
        p.error('Invalid hyperparameter range.')
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        p.error(
            'CUDA unavailable in this environment. Verify your terminal environment or explicitly choose --device cpu.',
        )
    #  fix seeds and choose the training device
    if args.d_model % args.num_heads != 0:
        p.error('d-model must be divisible by num-heads.')
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # create a randomly initialized Transformer model
    vocab = load_vocab(args.data_dir)
    model_config = dict(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        context_length=args.context_length,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = TransformerLanguageModel(**model_config).to(device)
    if args.output_dir.exists():
        p.error('Output exists; choose a fresh --output-dir.')
    if args.output_dir.resolve() == args.data_dir.resolve() or args.output_dir.resolve() in args.data_dir.resolve().parents:
        p.error('Output must not contain input data.')
    # create output files and save the configuration
    args.output_dir.mkdir(parents=True)
    pipeline_start = time.perf_counter()
    config = {}
    for name, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        config[name] = value
    config.update(
        model_config=model_config,
        precision='float32',
        optimizer='AdamW',
        num_workers=0,
        embedding_initialization='random trainable; PAD row fixed zero',
        output_weights_tied=False,
        loss='summed NLL / non-PAD targets',
        checkpoint_selection='minimum validation CE; strict improvement; patience epochs',
        determinism_note='Seeds and cuDNN deterministic settings fixed; exact cross-platform reproduction not guaranteed.',
    )
    save_json(args.output_dir / 'config.json', config)
    start = time.perf_counter()
    # load data. Only the training loader shuffles chunks
    datasets = {}
    summaries = {}
    loaders = {}
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    for split in ('train', 'validation', 'test'):
        dataset = ChunkDataset(args.data_dir, split, args.context_length)
        datasets[split] = dataset
        summaries[split] = dataset_summary(dataset)
        shuffle = split == 'train'
        loader_generator = shuffle_generator if shuffle else None
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=loader_generator,
            num_workers=0,
            pin_memory=device.type == 'cuda',
            collate_fn=collate_chunks,
        )
    data_loading_seconds = time.perf_counter() - start
    # set up the optimizer and record hardware
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    hardware = hardware_info()
    hardware['thread_note'] = 'DataLoader uses num_workers=0; PyTorch CPU thread settings recorded separately.'
    hardware['python_version'] = sys.version
    hardware.update(
        device=str(device),
        torch_version=torch.__version__,
        torch_cuda_runtime=torch.version.cuda,
        torch_cpu_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
    )
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        hardware.update(
            gpu_name=props.name,
            gpu_total_memory_bytes=props.total_memory,
            cudnn_version=torch.backends.cudnn.version(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    best = float('inf')
    best_epoch = 0
    bad_epochs = 0
    steps = 0
    telemetry_records = [{'phase': 'before_training', 'nvidia_smi': telemetry()}]
    checkpoint = args.output_dir / 'best_model.pt'
    training_seconds = validation_seconds = checkpoint_seconds = 0.0
    wall_start = time.perf_counter()
    # train, validate and save the best model each epoch
    for epoch in range(1, args.epochs + 1):
        train = train_one_epoch(model, loaders['train'], device, optimizer, args.grad_clip)
        val = evaluate(model, loaders['validation'], device)
        if train['target_tokens'] != summaries['train']['target_tokens'] or val['target_tokens'] != summaries['validation']['target_tokens']:
            raise AssertionError('Scored target count mismatch.')
        training_seconds += train['seconds']
        validation_seconds += val['seconds']
        steps += train['steps']
        row = dict(
            epoch=epoch,
            train_cross_entropy=train['cross_entropy'],
            train_perplexity=train['perplexity'],
            validation_cross_entropy=val['cross_entropy'],
            validation_perplexity=val['perplexity'],
            train_seconds=train['seconds'],
            validation_seconds=val['seconds'],
            train_target_tokens=train['target_tokens'],
            validation_target_tokens=val['target_tokens'],
            cumulative_optimizer_steps=steps,
            learning_rate=optimizer.param_groups[0]['lr'],
        )
        history.append(row)
        with (args.output_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        telemetry_records.append({'phase': f'after_epoch_{epoch}', 'nvidia_smi': telemetry()})
        if val['cross_entropy'] < best:
            best = val['cross_entropy']
            best_epoch = epoch
            bad_epochs = 0
            sync(device)
            start = time.perf_counter()
            model_state = {}
            for name, tensor in model.state_dict().items():
                model_state[name] = tensor.detach().cpu()
            torch.save(
                {
                    'model_state': model_state,
                    'optimizer_state': optimizer.state_dict(),
                    'model_config': model_config,
                    'config': config,
                    'epoch': epoch,
                    'validation_cross_entropy': best,
                    'vocabulary': vocab,
                    'vocab_sha256': hashlib.sha256((args.data_dir / 'vocab.txt').read_bytes()).hexdigest(),
                },
                checkpoint,
            )
            sync(device)
            checkpoint_seconds += time.perf_counter() - start
        else:
            bad_epochs += 1
        print(
            f"Epoch {epoch}: train CE={train['cross_entropy']:.4f}, val CE={val['cross_entropy']:.4f}, train={train['seconds']:.1f}s, val={val['seconds']:.1f}s, best={best_epoch}",
            flush=True,
        )
        if bad_epochs >= args.patience:
            print('Early stopping.', flush=True)
            break
    fit_wall_seconds = time.perf_counter() - wall_start
    training_peak = {'allocated_bytes': torch.cuda.max_memory_allocated(device), 'reserved_bytes': torch.cuda.max_memory_reserved(device)} if device.type == 'cuda' else None
    start = time.perf_counter()
    # reload the best checkpoint and test
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved['model_state'])
    del saved
    sync(device)
    reload_seconds = time.perf_counter() - start
    test = evaluate(model, loaders['test'], device)
    if test['target_tokens'] != summaries['test']['target_tokens']:
        raise AssertionError('Test target count mismatch.')
    start = time.perf_counter()
    plot_history(history, args.output_dir / 'loss_curves.png')
    plot_seconds = time.perf_counter() - start
    telemetry_records.append({'phase': 'after_test', 'nvidia_smi': telemetry()})
    # save comparison metrics and curves
    results = {}
    results['model'] = 'causal Transformer'
    results['config'] = config
    results['hardware'] = hardware
    results['datasets'] = summaries
    results['best_epoch'] = best_epoch
    results['validation'] = {'cross_entropy': best, 'perplexity': perplexity(best)}
    results['test'] = test
    results['model_size'] = {
        'total_parameters': sum((p.numel() for p in model.parameters())),
        'trainable_parameters': sum((p.numel() for p in model.parameters() if p.requires_grad)),
        'parameter_tensor_bytes': sum((p.numel() * p.element_size() for p in model.parameters())),
        'checkpoint_bytes': checkpoint.stat().st_size,
    }
    results['training_budget'] = {
        'max_epochs': args.epochs,
        'actual_epochs': len(history),
        'patience': args.patience,
        'optimizer_steps': steps,
        'processed_training_targets': summaries['train']['target_tokens'] * len(history),
    }
    results['timing'] = {
        'clock': 'time.perf_counter; CUDA synchronized at phase boundaries',
        'dataset_loading_seconds': data_loading_seconds,
        'training_epochs_seconds': training_seconds,
        'validation_epochs_seconds': validation_seconds,
        'checkpoint_save_seconds': checkpoint_seconds,
        'fit_and_selection_wall_seconds': fit_wall_seconds,
        'best_checkpoint_reload_seconds': reload_seconds,
        'test_seconds': test['seconds'],
        'plot_seconds': plot_seconds,
        'pipeline_elapsed_before_metrics_write_seconds': time.perf_counter() - pipeline_start,
        'epoch_timing_note': 'Includes batch collation/transfer and optimization or evaluation; excludes initial dataset loading and checkpoint writes.',
        'fit_wall_note': 'Includes train/validation, checkpoint saves, CSV writes and epoch telemetry.',
    }
    results['gpu_training_peak_memory'] = training_peak
    results['gpu_telemetry_snapshots'] = telemetry_records
    results['vocab_sha256'] = hashlib.sha256((args.data_dir / 'vocab.txt').read_bytes()).hexdigest()
    results['source_sha256'] = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in ('train_transformer.py', 'transformer_model.py', 'lm_data.py', 'tokenize_imdb.py')}
    results['inference_config'] = {
        'task': 'teacher-forced next-token scoring on the full test split',
        'device': str(device),
        'batch_size': args.batch_size,
        'context_length': args.context_length,
        'included': 'batch collation/transfer, forward pass and token-average cross-entropy calculation',
        'excluded': 'initial dataset loading, checkpoint loading and disk writes',
        'target_tokens': test['target_tokens'],
        'repetitions': 1,
    }
    results['inference_seconds'] = test['seconds']
    results['inference_tokens_per_second'] = test['target_tokens_per_second']
    save_json(args.output_dir / 'metrics.json', results)
    print(
        json.dumps(
            {
                'best_epoch': best_epoch,
                'validation': results['validation'],
                'test': test,
                'inference_tokens_per_second': results['inference_tokens_per_second'],
            },
            indent=2,
        ),
    )
    print(f'Results saved to {args.output_dir}')
if __name__ == '__main__':
    main()
