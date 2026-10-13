# Language Models, Embeddings, and Representations

## 1. Set Up

```bash
conda create -n dsa4213 python=3.11.16 pip -y
conda activate dsa4213
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Download and extract these files into the assignment folder:

- [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/):
  extract `aclImdb_v1.tar.gz` to create `aclImdb/`.
- [GloVe vectors](https://nlp.stanford.edu/projects/glove/):
  download `glove.6B.zip` and extract `glove.6B.100d.txt`.

## 2. Prepare the Language Modeling Data

```bash
# Clean HTML formatting, normalize text and convert reviews to lowercase.
python clean_imdb.py

# Split unlabeled reviews into training, validation and test sets.
python split_imdb.py

# Tokenize reviews, fix unusual characters and keep punctuation as tokens.
python tokenize_imdb.py

# Build a 15,000-token vocabulary using training data only.
python build_vocab.py

# Convert tokens to IDs and add BOS/EOS markers to each review.
python encode_imdb.py
```

This step splits the unlabeled data `aclImdb_cleaned/train/unsup` into 25,000 training, 5,000 validation and 5,000 test reviews.
The vocabulary contains four special tokens and the most frequent ordinary
tokens appearing at least twice in training.

Outputs:

- `aclImdb_cleaned/`: cleaned reviews with the original folder structure.
  `cleaning_summary.json` records cleaning statistics.
- `lm_splits/`: selected reviews in `train/`, `validation/` and `test/`.
- `lm_tokenized/`: tokenized reviews in the same three splits.
- `lm_vocab/`: `vocab.txt` defines the vocabulary and ID order.
  `train_word_counts.tsv` records all training token counts, including tokens outside the selected vocabulary. `vocabulary_report.txt` contains vocabulary details and coverage statistics for each split.
- `lm_encoded/`: one token-ID text file per review, organized by split,
  and a copy of `vocab.txt`. These files are used for LM training and evaluation.

## 3. Part I: Train and Test Four Language Models

```bash
python train_ngram.py
python train_rnn.py
python train_lstm.py
python train_transformer.py
```

Results are saved in `results_ngram/`, `results_rnn/`,
`results_lstm/` and `results_transformer/`.

## 4. Part II: Compare Input Embeddings

Reuse `results_transformer/` as the random, jointly learned embedding baseline.

Train Skip-gram vectors, then train a new Transformer with frozen Skip-gram embeddings:

```bash
python train_skipgram.py
python train_transformer_fixed.py --embedding-file results_skipgram/embedding.pt --output-dir results_transformer_skipgram
```

Use public GloVe vectors as the frozen embeddings to train another Transformer:

```bash
python prepare_glove.py --glove-file glove/glove.6B.100d.txt
python train_transformer_fixed.py --embedding-file results_glove/embedding.pt --output-dir results_transformer_glove
```

Both fixed settings freeze the entire input embedding table.
All other model parameters are trained from scratch.

## 5. Generate Text from Saved Models

Generation is separate from training and test evaluation:

```bash
python generate_saved_models.py --model ngram --results-dir results_ngram
python generate_saved_models.py --model rnn --results-dir results_rnn
python generate_saved_models.py --model lstm --results-dir results_lstm
python generate_saved_models.py --model transformer --results-dir results_transformer
python generate_saved_models.py --model transformer --results-dir results_transformer_skipgram
python generate_saved_models.py --model transformer --results-dir results_transformer_glove
```

All runs use the same seven prompts, temperature 0.8, top-k 50,
and a maximum of 150 new tokens. Generation stops at the first EOS.
Results are saved as `generation.json` in each model folder.

## 6. Part III: Sentiment Classification

Create the labeled splits once:

```bash
python split_sentiment_data.py
```

For sentiment classification, only labeled reviews are used.
The labeled reviews in `aclImdb_cleaned/train/pos/` and
`aclImdb_cleaned/train/neg/` are split into 20,000 training and
5,000 validation reviews. All 25,000 labeled reviews in
`aclImdb_cleaned/test/pos/` and `aclImdb_cleaned/test/neg/` form the test set.
These splits are saved in `sentiment_splits/`.
The unlabeled reviews used for language modeling are not used in this step.

Extract frozen LM features using combined mean and max pooling:

```bash
python extract_sentiment_features.py --model lstm --results-dir results_lstm
python extract_sentiment_features.py --model transformer --results-dir results_transformer
python extract_sentiment_features.py --model transformer --results-dir results_transformer_skipgram
python extract_sentiment_features.py --model transformer --results-dir results_transformer_glove
```

Train logistic regression and MLP classifiers with the same settings:

```bash
python train_sentiment_classifier.py --features-dir results_lstm/sentiment_features
python train_sentiment_classifier.py --features-dir results_transformer/sentiment_features
python train_sentiment_classifier.py --features-dir results_transformer_skipgram/sentiment_features
python train_sentiment_classifier.py --features-dir results_transformer_glove/sentiment_features
```

Summarize the four comparisons:

```bash
python summarize_sentiment.py
```

## 7. Output Files and Reproduction

- `results_*/metrics.json`: LM test scores, timing and model size.
- `results_*/config.json`: saved experiment settings.
- Neural model `history.csv` and `loss_curves.png`: training and validation losses.
- `results_*/generation.json`: generated text and decoding settings.
- `results_*/sentiment_features/`: extracted features and metadata.
- `results_*/sentiment_classifier/`: classifier scores, MLP loss curves and errors.
- `sentiment_comparison.csv` and `.json`: downstream comparison summary.

The commands use the script defaults and seed 42.
Generation uses seeds 42–48 for the seven prompts.
Exact scores and timing may vary across hardware and software environments.