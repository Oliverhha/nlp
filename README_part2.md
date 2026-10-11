# Part II: Transformer with frozen Skip-gram input embeddings

Use the dsa4213 environment and run commands yourself from the assignment folder.
No training, generation, correctness-check mode or smoke run was executed during
code creation. All code comments and docstrings are English.

## 1. Train the word vectors

```bash
python train_skipgram.py
```

The script streams the same tokenized training reviews used for Part I.
Skip-gram learns to predict nearby words, using Gensim with 100 dimensions,
window 5, negative sampling 5, minimum frequency 2, subsampling 0.001,
10 epochs, one CPU worker and seed 42. A stable hash function helps reproduce
initialization. Word2Vec's own vocabulary is built from training only; it may
contain more words than the fixed LM vocabulary. The LM vocabulary never changes.

Each review is an independent word sequence. Only train reviews fit vectors.
Validation/test encoded IDs are read afterward solely to report coverage.
The final matrix is aligned to the existing lm_encoded/vocab.txt row order.

Missing lexical words and UNK/BOS/EOS receive seeded Normal(0, 0.1) vectors.
PAD is zero. Learned vectors are copied without normalization or rescaling.
These choices are recorded, and all rows are later frozen. Special tokens are
excluded from lexical vocabulary coverage; PAD/BOS/EOS are excluded from token
coverage, while UNK counts as uncovered. Keep the same missing-row policy when
preparing the later GloVe matrix.

Files under results_skipgram/:

- skipgram_vectors.kv: learned keyed word vectors.
- embedding.pt: vocabulary-aligned matrix, exact vocabulary, coverage and metadata.
- config.json: Skip-gram settings.
- metrics.json: separate embedding training cost, vocabulary/token coverage and versions.
- missing_words.txt: lexical LM words without learned vectors.

The training cost is vocabulary construction plus Skip-gram training, including
streamed file reads. It is reported separately from Transformer training.

## 2. Train a fresh Transformer with the entire table frozen

```bash
python train_transformer_fixed.py
```

This loads results_skipgram/embedding.pt. It creates a fresh Transformer with
exactly the Part I architecture, copies only the input word-vector table, sets
embedding.weight.requires_grad_(False), and gives the optimizer only trainable
parameters. Missing words and special-token rows are also frozen.

The projection, positional embeddings, attention/feedforward layers,
normalization and output layer are randomly initialized and trained from scratch.
No Part I Transformer checkpoint is loaded for this new experiment.

Defaults match the existing Part I Transformer baseline: input dimension 100,
model width 128, two layers, four heads, feedforward width 512, dropout 0.2,
context 128, batch 16, AdamW learning rate 0.001, weight decay 0.01,
gradient clip 1.0, seed 42, max 60 epochs and patience 5. Compare the existing
Part I config against these settings before deliberately changing hyperparameters.

Training, per-epoch validation, best-checkpoint selection and final test remain
in this file. It saves results_transformer_skipgram/ with best_model.pt,
config.json, history.csv, loss_curves.png and metrics.json. Input embeddings are
included in the saved model. Total and trainable parameter counts are reported
separately because the frozen table remains part of the model.

Test CE/PPL ignores PAD. Inference means full-test-set next-token scoring with
real preceding tokens; the timing scope matches the Part I neural scripts.
The script neither generates text nor measures autoregressive generation speed.

Both training programs reject existing output directories. Choose a fresh
--output-dir to run a new experiment; do not overwrite previous results.

## 3. Generate independently from the newly saved Transformer

```bash
python generate_saved_models.py --model transformer --results-dir results_transformer_skipgram --max-new-tokens 150
```

The existing generation script loads one saved model and writes only
results_transformer_skipgram/generation.json. Its seven prompts, temperature
0.8, top-k 50, seeds and EOS stopping must match the Part I baseline comparison.
No training or test evaluation occurs here.

## Three-way comparison

Reuse results_transformer/ for the random trainable baseline. The Skip-gram run
is the first additional LM run. A later GloVe matrix can use the same frozen
Transformer trainer via --embedding-file and --output-dir; it must have the same
matrix/checkpoint format, dimension, vocabulary order and missing-row policy.
Preparing the public GloVe matrix is a separate later step.

Compare held-out CE/PPL, convergence/stability, generated examples and costs.
Interpret differences in both embedding source and trainability: random inputs
are trainable, Skip-gram/GloVe inputs are frozen, and public GloVe uses external
training data. Better performance of a variant is not assumed.
