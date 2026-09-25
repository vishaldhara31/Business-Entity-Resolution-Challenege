# How to run (SageMaker / any Linux box)

Needs Python 3.11+ and `pip install -r requirements.txt`. Recommended: a CPU instance with 16+ vCPU and 64 GB RAM
(for example ml.m5.4xlarge or ml.c5.9xlarge). Use `--n-jobs` to set the number of worker processes.
Data lives outside git. Layout expected by default:

```
Hackathon/Datasets/Train Datasets/train_source{1,2,3}.tsv, train_ground_truth.tsv
Hackathon/Datasets/test_source{1,2,3}.tsv
```
Override with `--train-dir`, `--test-dir`, `--work-dir`, `--output-dir`.

```bash
# 1. normalize the pools (writes work/pools/*.parquet)
python -m src.pipeline normalize --split train --n-jobs 15
python -m src.pipeline normalize --split test  --n-jobs 15

# 2. train the matcher: blocking + features + LightGBM + threshold tuning (writes work/matcher.txt, matcher_meta.json)
python -m src.pipeline train --n-entities 100000 --n-jobs 15

# 3. predict the test set (writes output/matching_results.tsv and output/candidate_pairs.tsv)
python -m src.pipeline predict --n-jobs 15

# 4. validate
python Hackathon/utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir Hackathon/Datasets
```

Useful options: `--k 20` (top-K per channel), `--maxdf 20000` (TF-IDF document-frequency cap; lower is faster),
`--shard 2000` (queries per worker task), `--only-country France --limit 5000` (quick partial test run).

## Design notes
- Blocking: word-level TF-IDF, cosine similarity (sparse dot product), separate name and address channels, per country; candidates = union of both channels' top-K.
- Matcher: LightGBM on country-agnostic pair features (string similarities, cosine scores, numeric-token and legal-suffix agreement). Country is only used to partition blocking; it is never a model feature, so unseen countries (France) are handled.
- Decision: keep candidates with predicted probability >= tuned threshold (tuned for per-entity macro F0.5 on held-out train entities).
