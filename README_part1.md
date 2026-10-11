# Part I: training, test inference, and separate generation

Use the dsa4213 environment from the assignment directory.

The four train_*.py scripts train on lm_encoded/train, use validation for
selection, save a model, then score lm_encoded/test. They do not generate text.
They report inference as next-token scoring on the entire shared test split.
Each target is scored using only preceding tokens; PAD is ignored in loss.
Token-average CE is total NLL divided by the total number of valid targets.
PPL is exp(CE). The corpus, vocabulary and context/chunk length are shared.

Context length is 128 by default. Each review is independent, and recurrent
state is reset at chunk boundaries. Each target appears exactly once. EOS is
scored, while PAD/BOS are excluded from prediction support.

Existing models do not need retraining after separating the scripts.
Run generation separately, for example:

```bash
python generate_saved_models.py --model lstm
```

Results are separated within each model's result directory:

- metrics.json: training, validation, test quality/cost, hardware and model size.
- generation.json: independently generated continuations and decoding settings.

Use the same prompts/settings and first-EOS stopping rule for all four models.
See README_generation.md for commands and output definitions.

Neural test timers include batch collation/transfer, forward pass and loss
calculation, excluding initial dataset and checkpoint loading. The n-gram test
timer includes test file loading, probability collection and loss calculation.
Document this timing-scope difference, CPU/GPU devices and neural batch sizes.
All models score the same test targets, but these measurements compare complete
implementations rather than isolating architecture or kernel performance.

The preserved historical training source hashes describe the original runs.
Cleaning JSON metadata and removing generation code does not change model weights
or test CE/PPL. Old mixed end-to-end totals were removed because generation time
cannot be subtracted exactly from them. Original files were backed up separately.
