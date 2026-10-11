# Part I TRANSFORMER

Random trainable 100d embeddings projected to width 128, learned positional
embeddings, two causal attention blocks with four heads and feedforward width
512, and a vocabulary output layer. TransformerEncoderLayer supplies attention,
feedforward, residual and normalization operations; the explicit causal mask
blocks future tokens. There is no encoder-decoder cross-attention.

All weights are trained from scratch. PAD/BOS cannot be predicted. Each chunk
starts with fresh context; token_states() exposes representations for Part III.

train_transformer.py trains, validates each epoch, saves the best validation checkpoint,
then evaluates the test split once. It contains no text generation or generation
benchmark. Architecture and optimization defaults can be read in its argparse
section; actual runs are documented by their saved config.json and metrics.json.

For a new training run only:

```bash
python train_transformer.py --output-dir results_transformer_new
```

No resume, correctness-check or smoke-run modes. Existing output directories
are rejected. Existing checkpoints do not need retraining after this edit.

Artifacts:

- config.json: training and evaluation configuration.
- best_model.pt: best model, optimizer state, model config and vocabulary.
- history.csv and loss_curves.png: training/validation losses and epoch timings.
- metrics.json: CE/PPL, hardware, training budget, model size and test inference.

Test inference uses the full common test split with real preceding tokens,
model.eval(), no gradients and non-PAD token-average loss. Its timer includes
batch collation/transfer, forward pass and loss calculation; it excludes initial
dataset/checkpoint loading and file writes. CUDA is synchronized at boundaries.
The timer is a single final test pass without a separate benchmark warmup.
Throughput is test target tokens / test seconds. This is not generation speed.

Training loss has dropout enabled; validation/test have it disabled. Temperature
snapshots are not continuous monitoring. Checkpoint size includes optimizer state;
parameter tensor bytes give a separate model-only measure.

Generate separately from the saved model:

```bash
python generate_saved_models.py --model transformer
```

Only results_transformer/generation.json is written. It records shared prompts,
sampling settings, EOS-stopped continuations, token IDs and actual lengths.
See README_generation.md for details. No training or test evaluation is performed.
