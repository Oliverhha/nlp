# Generate from one saved model

Run in the assignment directory using the dsa4213 environment. Choose one model:

```bash
python generate_saved_models.py --model ngram
python generate_saved_models.py --model rnn
python generate_saved_models.py --model lstm
python generate_saved_models.py --model transformer
```

Each command loads only the selected model and writes only its
`results_<model>/generation.json`. Repeating the command replaces that generation
file. To use a different trained run, add `--results-dir path_to_that_run`.

No training, validation, test evaluation, speed benchmark or metrics rewrite is
performed. Model weights are frozen during generation. The existing
`metrics.json`, checkpoints and loss curves remain unchanged.

The shared settings are seven prompts (four short and three with sentiment context), temperature 0.8, top-k 50, seeds 42--48,
and a maximum of 80 new tokens. Generation stops at the first EOS. It never
clears the context and starts a second text. The displayed continuation excludes
terminal EOS; token IDs and generated-token counts retain it. An early EOS is
reported even if the continuation is very short or empty. There is no cherry
picking or resampling to obtain longer text.

The JSON contains model/checkpoint information, generation settings, prompts,
continuations, token IDs, lengths and stop reasons. It contains no training or
test results. Training and test results belong in `metrics.json` instead.

The existing model files, lm_data.py, tokenize_imdb.py and train_ngram.py must be
beside this script. Importing NgramLM does not run its training main function.
The n-gram uses lm_encoded/vocab.txt, checked against its saved vocabulary hash.
Neural checkpoints contain their own vocabularies and model configurations.

The training scripts now report inference as full-test-set next-token scoring,
not text generation. See each model README for its timing scope. Keep generation
settings matched across models for the assignment comparison.

This script was written without executing training or generation.
