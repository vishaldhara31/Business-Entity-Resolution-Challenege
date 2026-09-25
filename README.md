# Business Entity Resolution Challenge

Match each Source 1 business to its records in Source 2 and Source 3 (some have no match).
Metric: per-entity F0.5, macro-averaged, singletons included. Deadline: 27 Sept 2026, 11:59 pm.

## Data findings (train)
- Source 1: 2.2M entities (1.32M US, 0.88M India). Train has only US and India; **test also has France**, so country is an open set and is never a model feature.
- Test scale: 1.73M Source 1 entities, about 4.9M Source 2 and 5.1M Source 3 rows (France is about 15%).
- About 5.6% of Source 1 entities have no match. Average matches per entity: about 1.7 in Source 2, about 1.8 in Source 3.
- Noise: domain-style names, native-script names (India), varying legal suffixes, abbreviated state names, reordered and truncated addresses, `<NULL>` and empty addresses. Postal codes are mostly absent, so they are not usable as keys.

## Approach (see `src/`)
1. **Normalize** names and addresses (accents, legal suffixes incl. French forms, state and street abbreviation expansion).
2. **Blocking:** word-level TF-IDF, cosine similarity via sparse dot product, separate name and address channels per country, union of top-K (K=20).
3. **Matcher:** LightGBM on country-agnostic pair features (rapidfuzz similarities, cosine scores, numeric-token and legal-suffix agreement, relative-to-group features).
4. **Decision:** keep candidates above a probability threshold tuned for macro F0.5.

## Results so far (5,000 train entities against the full train pools, held-out split)
| Stage | Value |
|---|---|
| Candidate recall (K=20, union of channels) | 94.8% (S2), 93.2% (S3) |
| Ceiling F0.5 with perfect classification of candidates | 0.979 |
| Matcher macro F0.5, held-out test | 0.925 |
| K=50 instead of K=20 | recall +2 points, F0.5 unchanged (0.926), 2.5x compute |

## Repo layout
- `src/norm.py`, `src/features.py`, `src/pipeline.py`: the pipeline (`normalize`, `train`, `predict`).
- `notebooks/`: experiments (01 blocking, 02 missed pairs and addresses, 03 TF-IDF blocking, 04 full-pool blocking, 05 matcher, 06 K=50 blocking, 07 K=50 matcher).
- `work/build_nb*.py`, `work/make_dev*.py`, `work/norm_full.py`: scripts that generate the notebooks and dev data.
- `RUN.md`: exact commands to reproduce on SageMaker. `requirements.txt`: pinned versions (Python 3.11+).

## Next
- Run the full pipeline on SageMaker (train with about 100k entities, then predict the full test set) and submit early.
- Check French predictions by eye (no labels for France).
- Improve the matcher and threshold; possible multilingual-embedding channel for native-script names (needs a GPU).

Datasets are not in the repo; place them under `Hackathon/Datasets/`.
