# Part II: public GloVe, frozen Transformer and separate generation

Use the dsa4213 environment. These instructions do not require retraining the
random baseline or the Skip-gram variant. No preparation, training or generation
was executed during code creation.

## Obtain the public vectors

Use the Stanford GloVe release page:
https://nlp.stanford.edu/projects/glove/

Select glove.6B.zip (Wikipedia 2014 + Gigaword 5, 6B tokens, uncased). The archive
contains several dimensions; extract only glove.6B.100d.txt. Put this text file
in the assignment directory or pass its actual path with --glove-file.
The archive download is about 822 MB. No automatic download is performed by the
preparation script.

## Prepare the matrix

```bash
# python prepare_glove.py --glove-file glove/glove.6B.100d.txt
python prepare_glove.py
```

This streams the public file and copies vectors for exact matches in the existing
15,000-token LM vocabulary. It does not train word vectors, change tokenization,
expand the vocabulary or learn from validation/test. The matrix shape is 15000
by 100, identical to the other settings.

Missing ordinary tokens and UNK/BOS/EOS retain Normal(0, 0.1) vectors with seed
42, matching the Skip-gram fallback policy. PAD is zero. Public vectors are not
rescaled or normalized. All input rows are later frozen, including missing and
special tokens. Exact lookup can leave contractions, misspellings and rare names
unmatched; the missing list and coverage rates document this limitation.

results_glove/ contains embedding.pt, config.json, metrics.json and
missing_words.txt. Coverage definitions match the Skip-gram preparation.
The recorded local time is file preparation, not the cost of training public
GloVe. Source-file hash and release information are recorded for attribution.

## Train and test the new language model

```bash
python train_transformer_fixed.py --embedding-file results_glove/embedding.pt --output-dir results_transformer_glove
```

The same trainer used for Skip-gram loads the matrix, freezes the whole input
table and initializes all other parameters from scratch. Architecture/data and
training settings remain the same: input 100d, internal width 128, two layers,
four heads, context 128, batch 16, dropout 0.2, AdamW 0.001, max 60 epochs,
patience 5 and seed 42. It does not load an existing LM checkpoint for training.

Training/validation/test results go to results_transformer_glove/metrics.json,
with best_model.pt, config.json, history.csv and loss_curves.png. Inference cost
means full-test-set next-token scoring, matching the other Transformer variants.

## Generate separately

```bash
python generate_saved_models.py --model transformer --results-dir results_transformer_glove --max-new-tokens 150
```

This writes only results_transformer_glove/generation.json using the same seven
prompts and EOS stopping as the previous variants. It does not train or test.

Both preparation and training reject existing output directories. Choose fresh
output paths for a new run; do not overwrite earlier results.

## Compare the three experiments

- results_transformer: random inputs trained jointly (existing baseline).
- results_transformer_skipgram: self-trained inputs kept fixed.
- results_transformer_glove: public inputs kept fixed.

Compare test CE/PPL, convergence/stability, costs and generated examples. State
coverage/fallback policy and GloVe's additional external data. A better public
result would not by itself isolate the effect of the GloVe algorithm; both data
source and embedding trainability differ across these settings.
