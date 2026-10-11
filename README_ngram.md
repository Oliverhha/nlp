# Interpolated n-gram language model

train_ngram.py counts unigram, bigram and trigram events from the training split,
selects interpolation weights using validation CE, saves ngram.pkl and scores the
test split. It does not generate text or benchmark generated continuations.

Unigram additive smoothing uses alpha=0.1 by default; bigram/trigram relative
frequencies fall back to lower order for unseen contexts. A validation grid uses
0.1 steps and strictly positive unigram weight. Context boundaries and prediction
support match the neural models through lm_data.py.

For a new training run only:

```bash
python train_ngram.py --output-dir results_ngram_new
```

Existing output directories are rejected. Existing results require no rerun.

metrics.json records quality, training/selection/test times, hardware, count
entries, checkpoint bytes, vocabulary/source hashes and test inference settings.
Test inference includes loading test files, collecting probabilities and computing
CE/PPL. Throughput is the actual test target count divided by that duration.
There is no separate timing warmup or repetition. Neural test timers exclude
initial data loading, so state this difference in the report.

Generation is independent:

```bash
python generate_saved_models.py --model ngram
```

This loads the existing pickle and writes results_ngram/generation.json only.
It stops at EOS or the common maximum length, without restarting the context.
