# Business Entity Resolution – Plan

## 0. Findings from the workspace
- Present: `Hackathon/Datasets/test_source{1,2,3}.tsv`, `Hackathon/Datasets/Train Datasets/train_{source1,2,3,ground_truth}.tsv`, `Hackathon/PS.txt`.
- Train scale: S1 ≈ 2.2M, S2 ≈ 5.0M, S3 ≈ 5.3M rows (about the same size as test). Ground truth has multi-match rows (e.g. 2 S2 + 3 S3 for one S1).
- Still missing: `utils/validate_submission.py`, `Documentation_template.md` (write our own validator if not supplied).
- Python is not available on PATH (Windows Store shim only) – install Python 3.11+ (or a conda env) before any work.
- Train data is huge: for development use a sample (e.g. 200k S1 entities + all their true matches + random distractors), then scale up.
- Test scale is large: S1 ≈ 1.73M, S2 ≈ 4.89M, S3 ≈ 5.08M rows. Countries: US, India, France (France unseen in train).
- Multi-script text exists (e.g. Devanagari names in S3), plus junk prefixes (`<< Team Ecole`), typos, uppercase/abbreviated addresses.
- Environment: disk is nearly full (`/tmp` write failed). Free space / use chunked processing; avoid huge intermediates.
- Constraint: final model MIT/Apache-2.0, ≤ 8B params. No external lookups/APIs/geocoders.

## 1. Approach overview
Pipeline: normalize → blocking (candidates) → pair features → GBDT matcher → per-S1 decision rules (precision-focused) → outputs.

Scale means pairwise scoring must be limited to a few (≈5–30) candidates per S1 record. All work must be vectorized/chunked, partitioned by country.

## 2. Data prep
1. Read with `sep="\t"`, `dtype=str`, `keep_default_na=False`.
2. Partition by country string (open set; no hard-coded list). France handled by generic rules.
3. Normalization (per-country configurable dictionaries, defaults generic):
   - Unicode NFKD, lowercase, strip accents (French), remove junk chars/prefixes (`<<`, etc.).
   - Transliterate Devanagari → Latin (rule-based/`indic-transliteration`-style table; no external service).
   - Names: expand/strip legal suffixes (inc, corp, ltd, llc, pvt, private, limited, sarl, sas, sci, s.a.s, gmbh…), `&`→`and`, sorted-token variant, phonetic key (Double Metaphone/Soundex).
   - Addresses: abbreviation map (rd/road, st/street, r./rue, bd/boulevard, ave…), extract house number, street name, city, state/region, PIN/ZIP/postal code, strip landmark phrases ("near …"), token-sort.
4. Check duplicates within S2/S3 (many-to-one to the same S1) – expect clusters.

## 3. Blocking / candidate generation (target recall ≥ 95%, ~10–30 cands/S1)
Union of several keys, within same country:
- B1: name token blocking (rare-token inverted index; skip tokens with high doc freq) via TF-IDF weighted.
- B2: char 3-gram TF-IDF (or MinHash/LSH) on normalized name, sparse top-k (chunked sparse matmul or `sparse_dot_topn`).
- B3: phonetic key of first significant name tokens + city/state.
- B4: address key: postal code + street/house number; city + first name token.
- B5 (optional): small embedding retrieval with an Apache/MIT multilingual model (e.g. `multilingual-e5-small`, MIT; ≤8B) + FAISS ANN, for transliteration/typo cases that lexical misses.
- Cap per-S1 list by blocking score; write `candidate_pairs.tsv` from the **final** list actually scored.
- Measure recall ceiling on the train validation split and tune caps.

## 4. Matching model
Features per (S1, Sx) pair:
- Name: Jaccard (tokens/3-grams), TF-IDF cosine, Jaro-Winkler, Levenshtein ratio (rapidfuzz), token-sort/partial ratio, first-token equal, initials/acronym match, phonetic equal, length diff, suffix-stripped equality, rare-token overlap.
- Address: same measures, house-number match, postal-code match/prefix, street match, city match, state match, missing-component flags.
- Combined: name×address interactions, source (S2/S3), country, candidate rank and score gap vs. next-best, number of S1 competing for the same record (reverse-rank).
- Embedding cosine (if B5 used).
Model: LightGBM/XGBoost (permissive license), trained on candidates from train blocking; positives from ground truth, negatives = hard negatives from blocking. Optional cross-encoder rerank of uncertain pairs (≤8B, MIT/Apache) if time allows.

Validation: hold out 20% of train S1 entities (group split by S1 id). Include a "leave-country-out" check (train on US, test on India) to estimate generalization to unseen France.

## 5. Decision rules (F0.5, macro over S1 incl. singletons)
- Threshold per entity: accept candidate if p ≥ t (tune t on validation to maximize macro F0.5; expect t≈0.6–0.8).
- Singletons matter: if best p < t → empty list. Tune separately per source, maybe per country.
- Mutual/uniqueness constraint: a S2/S3 record is assigned to at most one S1 (ground truth S1 is deduplicated) – keep the highest-scoring S1, drop others.
- Relative rule: include additional candidates only if p ≥ t and within δ of the top score.
- Optional transitive check: S2↔S3 agreement boosts confidence.

## 6. France / open-set handling
- No France training data: avoid country one-hot and country-specific learned features; use language-agnostic features.
- Add French normalization rules (rue/r., bd/boulevard, sarl/sas/sci/eurl, accents, postal codes 5 digits, département names).
- Optionally pseudo-label France: run the model, take high-confidence pairs, retrain (self-training), and check score distribution sanity.

## 7. Engineering
- Language: Python (pandas/polars, scikit-learn, rapidfuzz, lightgbm, scipy sparse, faiss-cpu optional).
- Process per country and per chunk of S1 (e.g. 50k) to bound RAM; store intermediates as parquet on a drive with free space.
- Repo layout (matches submission):
  ```
  code/business_entity_resolution/
    src/{normalize.py, blocking.py, features.py, train.py, predict.py, run_all.py}
    README.md
    requirements.txt
  output/{matching_results.tsv, candidate_pairs.tsv}
  ```
- Validate via `utils/validate_submission.py`; ensure one row per S1, no duplicates, only IDs existing in test, matches ⊆ candidates.

## 8. Milestones
1. Get train files + validator; EDA (noise types, match cardinality, singleton rate per country).
2. Normalization + blocking; measure recall/candidate size on train.
3. Feature + LightGBM baseline; tune threshold on validation.
4. Add embeddings/phonetic blocking, uniqueness constraint; iterate on error analysis.
5. Run on full test, write both TSVs, validate.
6. Package zip: output/, code/, filled Documentation_template.md.

## 9. Risks
- Runtime/memory on ~11M records → aggressive blocking, chunking, rare-token pruning.
- Disk space → clean up before running.
- Unseen France → generic features, validation by leave-country-out, self-training.
- Non-Latin scripts → transliteration + char-level features + multilingual embeddings.
- Precision → conservative thresholds; singletons worth full credit.
