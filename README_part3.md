# Part III: Context-aware frozen LM representations for IMDb sentiment

## Experiment

Compare four saved language models on labeled IMDb sentiment classification:
LSTM, Transformer with jointly trained input embeddings, Transformer with frozen
Skip-gram embeddings, and Transformer with frozen public GloVe embeddings.
All language models remain frozen. No Part I/II training or result files change.

The document representation concatenates mean and max pooling (256d).
For each language model, train the same MLP family and a logistic regression
baseline. There is no mean-only experiment or pooling selection option.

The comparison asks whether lower LM perplexity also means better sentiment
features, and whether representation ranking depends on the classifier.
Mean+max is a motivated choice, not a promise of higher accuracy.

## Environment

Use dsa4213. Dependencies: torch, numpy, scikit-learn, matplotlib, joblib,
threadpoolctl. These dependencies are already present in the current environment.
For a fresh environment, install the appropriate torch build and these packages.

```bash
conda activate dsa4213
cd /home/plj/DSA4213/assignment
```

No feature extraction, classifier training or trial runs were executed when
these scripts were written or revised.

## 1. Prepare one shared labeled dataset

Run this separate program once, before loading any language model:

```bash
python split_sentiment_data.py
```

Default source: aclImdb_cleaned. The script copies the already cleaned labeled
review TXT files, leaving the source intact. With stratified seed 42, it splits
the official labeled training reviews into 20,000 train and 5,000 validation
reviews. All 25,000 official test reviews retain their official test membership.
Unlabeled reviews are not used in this classifier dataset.

The prepared directory is shared by every model:

```text
sentiment_splits/
    train/
        neg/          # 10000 reviews
        pos/          # 10000 reviews
    validation/
        neg/          # 2500 reviews
        pos/          # 2500 reviews
    test/
        neg/          # 12500 reviews
        pos/          # 12500 reviews
    manifest.json
```

The manifest records row order, labels, relative review paths, original source
paths, content hashes, counts and split seed. It is written after all copies
finish. Re-running the splitter refuses to overwrite an existing directory.
Use --input-dir and --output-dir for different paths. If a custom output is
selected, pass that same directory to every extractor with --data-dir.

## 2. Extract mean+max features

Run sequentially, loading only one checkpoint at a time:

```bash
python extract_sentiment_features.py --model lstm --results-dir results_lstm
python extract_sentiment_features.py --model transformer --results-dir results_transformer
python extract_sentiment_features.py --model transformer --results-dir results_transformer_skipgram
python extract_sentiment_features.py --model transformer --results-dir results_transformer_glove
```

Default input: sentiment_splits. The extractor only reads this prepared dataset
and its manifest.json; it never creates, changes or randomly divides dataset
splits. It stops with instructions to run split_sentiment_data.py if the prepared
manifest is missing. Every model reads the same review files and sample order.
Reuse tokenize_imdb.tokenize and the vocabulary inside best_model.pt. Do not
build a vocabulary from labeled data. The dataset copies have already passed
through the existing cleaning/lowercasing pipeline.

### LSTM: preserve recurrent state

Process each review from BOS through its last actual token. Compute in chunks
of 128 positions, passing both hidden and cell states from one chunk to the
next. Reset only when beginning another review. The computation chunk length is
not a hard limit on recurrent history. This model is unidirectional: each state
still depends only on its current token and preceding context.

Reviews are sorted by length for batching and features are written back into
the original manifest order. Active reviews form a prefix of each sorted batch.
Completed reviews leave the batch, along with their hidden/cell states. Right
padding appears only in a review's final chunk; its padded ending state is never
carried into another chunk or review. Valid earlier token states are unaffected
by later right padding in this unidirectional LSTM.

The extractor calls the existing model.embedding and model.lstm with a hidden
state argument. It does not change lstm_model.py or the saved weights. eval mode
disables dropout. The LSTM was originally trained with independently reset LM
chunks; carrying state over longer sequences changes its inference usage and
does not guarantee better performance or unlimited effective memory.

### Transformer: overlap finite windows

Default window length 128, stride 64. Positions restart within each window,
using only the trained position embedding table. For example, windows cover
input positions 1-128, 65-192, 129-256, and so on, including BOS once at the
review's beginning. The first window contributes all actual token states.
Later windows contribute only newly covered token states: their overlapping
prefix supplies context but is excluded from pooling. Each actual review token
contributes exactly once; no token is deleted or counted twice.

The first window has short context near the document beginning. In later windows,
new tokens have at least 64 preceding input positions with the default settings,
and at most 127. Overlap reduces artificial boundary fragmentation but does not
provide whole-review attention or a recurrent memory across windows. Computation
is roughly twice nonoverlapping extraction for long reviews.

Use --stride to change Transformer overlap. It must be positive and smaller
than --chunk-length, and must match across all three Transformer variants.
The stride argument is ignored for LSTM. Default extraction device is CUDA,
batch size 32, FP32, CPU threads 4. --device cpu is supported. If reducing
--batch-size for resource reasons, use the same value when comparing extraction
costs. Increasing window length beyond the trained context is rejected.

### Pooling and output

Take last-layer contextual states, bypassing the vocabulary output projection.
Exclude PAD/BOS/EOS from pooling; UNK represents a real review token and remains.
EOS is not appended as an input. Sum valid states across the full review and
divide by its total actual token count. Compute a coordinate-wise maximum of
the same valid states. Padding is masked to negative infinity for max pooling.

Concatenate the 128d token-weighted mean and the 128d maximum, producing a
single 256d feature vector. Computing the mean is part of this concatenation;
it is not saved or evaluated as a separate representation.

Mean summarizes overall information; max may retain strong feature responses
that are diluted by a long review, but may also amplify noise. Pooling cannot
restore context that the LM never saw.

Outputs for each LM:

```text
results_lstm/sentiment_features/
    train.npz
    validation.npz
    test.npz
    metadata.json
```

NPZ contains features, labels, review_ids, token_counts. Metadata records schema
version 2, checkpoint/vocabulary/split hashes, context policy, pooling, dimension,
device, UNK coverage and timing.
The extraction timer includes batching, transfers, LM hidden states, pooling
and CPU accumulation; excludes data preparation, model loading and saving.
Data preparation is timed separately. First-batch initialization is included.

These per-model NPZ files contain derived feature vectors and matching labels,
not new splits or copies of the review TXT dataset. Different models need
different feature files, while sharing the one prepared dataset directory.

## 3. Train classifiers

```bash
python train_sentiment_classifier.py --features-dir results_lstm/sentiment_features
python train_sentiment_classifier.py --features-dir results_transformer/sentiment_features
python train_sentiment_classifier.py --features-dir results_transformer_skipgram/sentiment_features
python train_sentiment_classifier.py --features-dir results_transformer_glove/sentiment_features
```

The MLP is Linear(256,128), ReLU, Dropout, Linear(128,2), with 33,154 parameters
for each of the four language-model representations.

StandardScaler is fitted only on classification training features and applied
unchanged to validation/test. MLP uses cross-entropy, Adam, batch size 128,
learning rate 0.001, dropout 0.2, weight decay 0.0001, maximum 50 epochs,
patience 5 and seed 42. Select best epoch by validation
macro-F1, breaking exact ties using lower validation CE. Training CE uses active
dropout; validation disables it.

Every representation uses the same fixed hyperparameters and seed. There is no
hyperparameter search or multi-seed loop. Each command trains one MLP and one
logistic classifier. Different representations may reach their best validation
epoch at different times under the same early-stopping rule. Report the single
run's accuracy and macro-F1, without a seed mean or standard deviation.

Logistic regression: L2, lbfgs, max_iter=2000, fixed C=1.0 and seed 42.
Run once per feature set. A convergence warning raises
an error rather than silently accepting unconverged optimization.

All classifier choices are fixed before test evaluation. No train+validation
refit. Default classifier device is CPU, with four torch and BLAS threads.
--device cuda optionally uses CUDA for MLP; logistic remains CPU. Keep device
and procedure consistent across representation comparisons.

Outputs are separated by LM. The fixed-protocol directory is separate from any
previous tuning results, which remain intact:

```text
results_lstm/sentiment_classifier_fixed/
    config.json
    scaler.joblib
    metrics.json
    logistic/
        classifier.joblib
        metrics.json
        test_predictions.csv
        error_examples.json
    mlp/
        best_classifier.pt
        metrics.json
        history.csv
        loss_curves.png
        test_predictions.csv
        error_examples.json
```

Metrics: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix
(rows=true, columns=predicted; negative then positive), model size/parameter count
and timing. Saved models include the classifier and standardization statistics.

Test classifier timing uses all test feature vectors after an excluded warm-up
batch, with CUDA synchronization where needed. It excludes LM extraction,
standardization, metrics and saving. Logistic uses one full-matrix call, MLP
uses batches. Report those timing scopes instead of calling classifier-only
time end-to-end text inference.

Each error file contains the ten most confident false positives, ten most
confident false negatives, and ten predictions nearest probability 0.5, with
cleaned text. These diagnostic selections are not random samples. Read them to
identify concrete failure patterns; do not use them to retune on the same test.

## 4. Collect results

After classifier experiments finish for all four LMs:

```bash
python summarize_sentiment.py
```

This reads each results_*/sentiment_classifier_fixed directory and writes
sentiment_comparison.csv and sentiment_comparison.json directly in the assignment
directory, without creating another result directory. The table has eight rows:
four LMs and two classifier families, all using mean+max pooling. Original LM
test PPL is included. Shared data/classifier settings must match, and Transformer
variants must share the same context policy.
LSTM and Transformer context policies are intentionally recorded separately.
The summarizer requires the same fixed settings and seed across representations,
and refuses to mix old tuning results with the fixed protocol.

The summary files are rewritten each time. This command trains nothing.

## Migration and interpretation

The old independent-chunk mean features are not interchangeable with these
features. If an old extraction directory already exists, leave it intact and
choose a fresh output root, for example:

```bash
python extract_sentiment_features.py --model lstm --results-dir results_lstm --output-dir results_lstm/sentiment_features_rerun
python train_sentiment_classifier.py --features-dir results_lstm/sentiment_features_rerun --output-dir results_lstm/sentiment_classifier_fixed_rerun
```

Repeat the corresponding paths for other LMs. All extractors read the same
sentiment_splits directory. The old sentiment_data/splits.json is no longer
used; only split_sentiment_data.py performs data splitting. To adopt this pipeline,
extract all four representations from the newly prepared common directory rather
than mixing old and new feature metadata. Feature extraction and classifier training refuse
to overwrite existing output directories. For a separate classifier rerun,
choose --output-dir explicitly inside each LM result directory. When the same
custom classifier subdirectory is used for all LMs, select it with
--classifier-subdir when summarizing. --results-root selects the parent directory
containing the four LM result directories.

Compare LMs within one classifier family first. Examine whether LM PPL ranking
agrees with downstream ranking.
Fixing seed 42 makes the random initialization/shuffle reproducible in the same
setup; exact numerical equality also depends on software, hardware and backend
operations. A single seed does not measure variability across classifier runs.
Recorded multiple seeds can also be reproduced, but are not used in this protocol.
LSTM has recurrent history whereas Transformer has finite-window history; do
not attribute their downstream differences solely to architecture. No result
establishes that these context changes improved accuracy without an appropriate
baseline experiment. They also do not fix the earlier LM training context.

Labeled and unlabeled IMDb are from the same domain with separate official
samples. This is not cross-domain transfer. Mean/max aggregation discards the
order of the contextual states; neither it nor a bigger classifier guarantees
correct handling of negation, sarcasm or mixed sentiment.

References and attribution:

- IMDb dataset: https://ai.stanford.edu/~amaas/data/sentiment/
- SentEval uses classifiers on fixed representations:
  https://aclanthology.org/L18-1269/
- ULMFiT uses last/mean/max concatenation. This experiment borrows the combination
  pooling idea but omits last-state pooling and does not reproduce its fine-tuning:
  https://aclanthology.org/P18-1031/
- Transformer-XL discusses context fragmentation. We do not implement its
  recurrent attention or positional encoding:
  https://aclanthology.org/P19-1285/

Acknowledge substantial coding-assistant help in the report/project README.
