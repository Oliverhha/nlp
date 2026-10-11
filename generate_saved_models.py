# generate text from ONE saved model, without training or test evaluation.

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from lm_data import BOS, EOS, UNK, load_vocab
from tokenize_imdb import tokenize
from rnn_model import RNNLanguageModel
from lstm_model import LSTMLanguageModel
from transformer_model import TransformerLanguageModel

from train_ngram import NgramLM


PROMPTS = [
    'the movie was',
    'i expected this film to',
    'the acting and the story',
    'although the beginning was',
    'i really enjoyed this film . the characters were believable and the performances were convincing .',
    'i was disappointed by this film . the plot was confusing and the dialogue felt unnatural .',
    'the acting was convincing , but the story moved too slowly .',
]
MODEL_CLASSES = {'rnn': RNNLanguageModel, 'lstm': LSTMLanguageModel,
                 'transformer': TransformerLanguageModel}


@torch.inference_mode()
def generate_neural(model, ids, device, args, seed, context_length):
    rng = torch.Generator(device=device).manual_seed(seed)
    history = list(ids)
    generated = []
    for _ in range(args.max_new_tokens):
        inputs = torch.tensor([history[-context_length:]],
                              dtype=torch.long, device=device)
        logits = model(inputs)[0, -1] / args.temperature
        values, indices = torch.topk(logits, min(args.top_k, logits.numel() - 2))
        sampled = torch.multinomial(torch.softmax(values, dim=-1), 1, generator=rng)
        token = int(indices[sampled].item())
        generated.append(token)
        if token == EOS:
            break
        history.append(token)
    return generated


def generate_ngram(model, ids, args, seed, context_length):
    rng = np.random.default_rng(seed)
    history = list(ids)
    generated = []
    for _ in range(args.max_new_tokens):
        probabilities = model.distribution(history[-context_length:])
        indices = np.flatnonzero(probabilities > 0)
        if args.top_k < len(indices):
            indices = indices[np.argsort(probabilities[indices])[-args.top_k:]]
        logits = np.log(probabilities[indices]) / args.temperature
        values = np.exp(logits - logits.max())
        token = int(rng.choice(indices, p=values / values.sum()))
        generated.append(token)
        if token == EOS:
            break
        history.append(token)
    return generated


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['ngram', 'rnn', 'lstm', 'transformer'])
    parser.add_argument('--results-dir', type=Path, help='Defaults to results_<model> beside this script.')
    parser.add_argument('--data-dir', type=Path, default=base / 'lm_encoded')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.max_new_tokens, args.top_k, args.cpu_threads) < 1 or args.temperature <= 0:
        parser.error('Token limit, top-k, threads and temperature must be positive.')
    source = args.results_dir or base / f'results_{args.model}'
    device = torch.device('cpu' if args.model == 'ngram' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; activate your training environment or select --device cpu.')
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f'Loading saved {args.model}', flush=True)
    if args.model == 'ngram':
        metrics = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
        vocab = load_vocab(args.data_dir)
        vocab_hash = hashlib.sha256((args.data_dir / 'vocab.txt').read_bytes()).hexdigest()
        if vocab_hash != metrics['vocab_sha256']:
            raise ValueError('Vocabulary differs from the trained n-gram vocabulary.')
        checkpoint = source / 'ngram.pkl'
        with checkpoint.open('rb') as stream:
            model = pickle.load(stream)
        context_length = metrics['context_length']
    else:
        checkpoint = source / 'best_model.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        vocab = saved['vocabulary']
        vocab_hash = saved['vocab_sha256']
        model = MODEL_CLASSES[args.model](**saved['model_config'])
        model.load_state_dict(saved['model_state'])
        model.to(device).eval()
        context_length = saved['config']['context_length']
        del saved

    # generate from the same prompts, stopping at EOS.
    word_to_id = {word: index for index, word in enumerate(vocab)}
    texts = []
    for index, prompt in enumerate(PROMPTS):
        ids = [BOS] + [word_to_id.get(word, UNK) for word in tokenize(prompt)]
        seed = args.seed + index
        if args.model == 'ngram':
            out = generate_ngram(model, ids, args, seed, context_length)
        else:
            out = generate_neural(model, ids, device, args, seed, context_length)
        ended = out[-1] == EOS
        visible = out[:-1] if ended else out
        texts.append({'prompt': prompt, 'prompt_ids': ids, 'seed': seed,
                      'continuation': ' '.join(vocab[token] for token in visible),
                      'token_ids': out, 'text_tokens': len(visible),
                      'generated_tokens_including_eos': len(out),
                      'stop_reason': 'eos' if ended else 'max_new_tokens'})

    results = {
        'model': args.model, 'checkpoint': str(checkpoint.resolve()),
        'device': str(device), 'vocab_sha256': vocab_hash,
        'generation_config': {
            'max_new_tokens': args.max_new_tokens, 'context_length': context_length,
            'temperature': args.temperature, 'top_k': args.top_k,
            'sampling': 'top-k temperature-adjusted multinomial sampling',
            'eos_policy': 'stop at first EOS',
            'rendering': 'space-separated tokens; terminal EOS hidden in text but retained in token_ids',
        },
        'generation': texts,
    }
    output = source / 'generation.json'
    output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {output}.')


if __name__ == '__main__':
    main()
