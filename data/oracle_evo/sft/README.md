# Oracle EVOBC Synthetic Corpora

Two CPU-generated multitask pseudo-parallel corpora are available.

## Files

- `synthetic_multitask.jsonl`
  Broad-coverage baseline.
  Approximate stats in the current snapshot:
  `50,000` rows, mean input length `24.9`, median `22`, mean `[UNK]` count `0.537`.

- `synthetic_multitask_short24.jsonl`
  Short-text-oriented variant.
  Approximate stats in the current snapshot:
  `50,000` rows, mean input length `16.29`, median `16`, mean `[UNK]` count `0.341`.

## Recommendation

Use `synthetic_multitask_short24.jsonl` as the default synthetic source for the
first oracle portability run. Keep `synthetic_multitask.jsonl` as a broader
coverage control in case the short version turns out to be too narrow.

