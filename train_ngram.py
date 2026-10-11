# train interpolated trigram LM, choose weights on validation, evaluate test.

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import pickle
import platform
import random
import time
import numpy as np
from lm_data import PAD, BOS, load_vocab, documents, events

class NgramLM:

    def __init__(self, vocab_size, alpha=0.1):
        self.v = vocab_size
        self.alpha = alpha
        self.uni = Counter()
        self.bi = Counter()
        self.tri = Counter()
        self.bi_context = Counter()
        self.tri_context = Counter()
        self.weights = (0.2, 0.3, 0.5)

    def fit(self, data_dir, context_length, loaded_documents=None):
        if loaded_documents is None:
            loaded_documents = documents(data_dir, 'train', self.v)
        self.training_documents = 0
        for ids in loaded_documents:
            self.training_documents += 1
            for previous_previous, previous, target in events(ids, context_length):
                self.uni[target] += 1
                # pack a pair of token IDs into one integer counter key
                bigram_key = previous * self.v + target
                self.bi[bigram_key] += 1
                self.bi_context[previous] += 1
                if previous_previous is not None:
                    context_key = previous_previous * self.v + previous
                    trigram_key = context_key * self.v + target
                    self.tri[trigram_key] += 1
                    self.tri_context[context_key] += 1
        self.total = sum(self.uni.values())

    def components(self, previous_previous, previous, target):
        # return unigram, bigram and trigram probabilities for one target.
        # PAD and BOS are excluded from the prediction vocabulary
        prediction_vocab_size = self.v - 2
        unigram = (self.uni[target] + self.alpha) / (self.total + self.alpha * prediction_vocab_size)
        bigram_count = self.bi_context[previous]
        if bigram_count > 0:
            bigram_key = previous * self.v + target
            bigram = self.bi[bigram_key] / bigram_count
        else:
            bigram = unigram
        # if the context is missing, fall back to the lower order distribution
        trigram = bigram
        if previous_previous is not None:
            context_key = previous_previous * self.v + previous
            context_count = self.tri_context[context_key]
            if context_count > 0:
                trigram_key = context_key * self.v + target
                trigram = self.tri[trigram_key] / context_count
        return (unigram, bigram, trigram)

    def distribution(self, history):
        previous = history[-1]
        previous_previous = None
        if len(history) > 1:
            previous_previous = history[-2]
        probabilities = np.zeros(self.v, dtype=np.float64)
        for target in range(self.v):
            if target == PAD or target == BOS:
                continue
            components = self.components(previous_previous, previous, target)
            probabilities[target] = np.dot(self.weights, components)
        return probabilities

def collect_components(model, data_dir, split, context_length, loaded_documents=None):
    if loaded_documents is None:
        loaded_documents = documents(data_dir, split, model.v)
    values = []
    for ids in loaded_documents:
        for previous_previous, previous, target in events(ids, context_length):
            probabilities = model.components(previous_previous, previous, target)
            values.append(probabilities)
    if not values:
        raise ValueError('No evaluation targets.')
    return np.asarray(values, dtype=np.float64)

def quality(components, weights):
    # weighted sum of the three probabilities for every target
    probabilities = components @ np.asarray(weights)
    negative_log_probabilities = -np.log(probabilities)
    ce = float(negative_log_probabilities.mean())
    return {'cross_entropy': ce, 'perplexity': math.exp(ce), 'target_tokens': len(components)}


def hardware_info():
    info = {
        'device': 'CPU',
        'platform': platform.platform(),
        'python_version': sys.version,
        'logical_cpus': os.cpu_count(),
        'cpu_model': platform.processor() or None,
        'training_counting_workers': 1
    }
    cpu_path = Path('/proc/cpuinfo')
    if cpu_path.exists():
        for line in cpu_path.read_text().splitlines():
            if line.startswith('model name'):
                info['cpu_model'] = line.split(':', 1)[1].strip()
                break
    mem_path = Path('/proc/meminfo')
    if mem_path.exists():
        for line in mem_path.read_text().splitlines():
            if line.startswith('MemTotal:'):
                info['system_ram_bytes'] = int(line.split()[1]) * 1024
    if hasattr(os, 'sched_getaffinity'):
        info['available_logical_cpus'] = len(os.sched_getaffinity(0))
    try:
        from threadpoolctl import threadpool_info
        info['numpy_threadpools'] = threadpool_info()
    except ImportError:
        info['numpy_threadpools'] = None
    return info

def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=base / 'lm_encoded')
    p.add_argument('--output-dir', type=Path, default=base / 'results_ngram')
    p.add_argument('--context-length', type=int, default=128)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    pipeline_start = time.perf_counter()
    if args.alpha <= 0 or args.context_length < 1:
        p.error('alpha and context length must be positive.')
    if args.output_dir.exists():
        p.error('Output exists; choose a fresh --output-dir.')
    # load fixed vocabulary and count training n-grams
    random.seed(args.seed)
    vocab = load_vocab(args.data_dir)
    model = NgramLM(len(vocab), args.alpha)
    train_documents = list(documents(args.data_dir, 'train', model.v))
    start = time.perf_counter()
    model.fit(args.data_dir, args.context_length, loaded_documents=train_documents)
    training_seconds = time.perf_counter() - start
    del train_documents
    print(f'Counting finished: {model.total} targets, {training_seconds:.2f}s', flush=True)
    validation_documents = list(documents(args.data_dir, 'validation', model.v))
    start = time.perf_counter()
    # calculate validation probabilities
    validation = collect_components(model, args.data_dir, 'validation', args.context_length,
                                    loaded_documents=validation_documents)
    validation_components_seconds = time.perf_counter() - start
    del validation_documents
    start = time.perf_counter()
    # select weights using validation CE
    search_results = []
    best = None
    for u in range(1, 11):
        for b in range(0, 11 - u):
            weights = (u / 10, b / 10, (10 - u - b) / 10)
            result = quality(validation, weights)
            search_results.append({'weights': weights, **result})
            if best is None or result['cross_entropy'] < best[0]:
                best = (result['cross_entropy'], weights)
    validation_search_seconds = time.perf_counter() - start
    model.weights = best[1]
    validation_result = quality(validation, model.weights)
    del validation
    test_documents = list(documents(args.data_dir, 'test', model.v))
    start = time.perf_counter()
    # evaluate test with the selected weights.
    test = collect_components(model, args.data_dir, 'test', args.context_length,
                              loaded_documents=test_documents)
    test_result = quality(test, model.weights)
    test_evaluation_seconds = time.perf_counter() - start
    test_result['seconds'] = test_evaluation_seconds
    test_result['target_tokens_per_second'] = test_result['target_tokens'] / test_evaluation_seconds
    del test, test_documents
    # save model
    args.output_dir.mkdir(parents=True)
    checkpoint = args.output_dir / 'ngram.pkl'
    start = time.perf_counter()
    with checkpoint.open('wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    serialization_seconds = time.perf_counter() - start
    results = {}
    results['validation'] = validation_result
    results['test'] = test_result
    results['weights'] = model.weights
    results['training_seconds_counting'] = training_seconds
    results['checkpoint_bytes'] = checkpoint.stat().st_size
    results['stored_ngram_entries'] = len(model.uni) + len(model.bi) + len(model.tri)
    results['seed'] = args.seed
    results['context_length'] = args.context_length
    results['alpha'] = args.alpha
    results['vocab_sha256'] = hashlib.sha256((args.data_dir / 'vocab.txt').read_bytes()).hexdigest()
    results['hardware'] = hardware_info()
    results['numpy_version'] = np.__version__
    results['timing'] = {
        'clock': 'time.perf_counter (wall clock)',
        'counting_training_seconds': training_seconds,
        'validation_components_seconds': validation_components_seconds,
        'validation_weight_search_seconds': validation_search_seconds,
        'fit_and_selection_seconds': training_seconds + validation_components_seconds + validation_search_seconds,
        'test_evaluation_seconds': test_evaluation_seconds,
        'checkpoint_serialization_seconds': serialization_seconds,
        'pipeline_elapsed_before_metrics_write_seconds': time.perf_counter() - pipeline_start,
    }
    results['training_budget'] = {
        'training_documents': model.training_documents,
        'training_target_tokens': model.total,
        'training_passes': 1,
        'vocabulary_size': model.v,
        'prediction_support_size': model.v - 2,
        'validation_target_tokens': validation_result['target_tokens'],
        'test_target_tokens': test_result['target_tokens'],
        'weight_candidates': len(search_results),
        'weight_grid_step': 0.1,
        'selection_metric': 'validation token-average cross-entropy',
        'alpha_tuned': False,
    }
    results['validation_weight_search'] = search_results
    results['inference_config'] = {
        'task': 'teacher-forced next-token scoring on the full test split',
        'device': 'cpu',
        'context_length': args.context_length,
        'target_tokens': test_result['target_tokens'],
        'repetitions': 1,
        'included': 'probability component collection and CE/PPL calculation',
        'excluded': 'initial test dataset loading, model training, validation selection, checkpoint serialization and disk writes',
    }
    results['inference_seconds'] = test_evaluation_seconds
    results['inference_tokens_per_second'] = test_result['target_tokens_per_second']
    results['model_size'] = {
        'checkpoint_bytes': checkpoint.stat().st_size,
        'unigram_entries': len(model.uni),
        'bigram_entries': len(model.bi),
        'trigram_entries': len(model.tri),
        'bigram_context_entries': len(model.bi_context),
        'trigram_context_entries': len(model.tri_context),
    }
    results['source_sha256'] = {}
    for name in ('train_ngram.py', 'lm_data.py', 'tokenize_imdb.py'):
        content = (base / name).read_bytes()
        results['source_sha256'][name] = hashlib.sha256(content).hexdigest()
    (args.output_dir / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    print(
        json.dumps(
            {
                'validation': validation_result,
                'test': test_result,
                'weights': model.weights,
            },
            indent=2,
        ),
    )
    print(f'Results saved to {args.output_dir}')
if __name__ == '__main__':
    main()
