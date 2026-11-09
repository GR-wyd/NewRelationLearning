# Spatial Biology and Gene Regulation Discovery Demo

This is a small, reproducible prototype for discovering candidate concept combinations from spatial biology or gene regulation literature. It uses up to 1,000 Europe PMC abstracts from 2022–2023, Qwen3 concept and claim extraction, an MPNet embedding graph, three candidate-generation baselines, and a time-separated 2024 evidence search. The optional 2025 run is a frozen-candidate retrospective test and must not be used to tune the pipeline.

The prototype does not train a relation encoder or HDBSCAN prototypes. The 2024 corpus is independent validation evidence, not proof that a relation is a novel discovery. Since local and API models may have seen papers from these years during pretraining, describe this as time-separated retrospective literature validation, not proof of model ignorance.

## Requirements

Use the Python interpreter that has the local model dependencies. Install PyTorch using a build compatible with the machine's CUDA setup, then install:

```bash
python -m pip install -r requirements.txt
```

The default model paths point to `/gpfs/share/home/2201110046/models/qwen3/14B` and `/gpfs/share/home/2201110046/models/mpnet`. Override them with `--qwen-model`, `--mpnet-model`, or `QWEN_MODEL_PATH` / `MPNET_MODEL_PATH`.

## Run

Run commands from this directory. Keep inputs, thresholds, and outputs under a run directory. The historical corpus is the union of spatial biology and gene regulation papers from 2022–2023. Do not put 2024 or 2025 records in `history_2022_2023.jsonl`.

Create the data and results directories:

```bash
mkdir -p runs/demo/data runs/demo/results
```

### 1. Fetch abstracts

Fetch and freeze the historical corpus and the 2024 validation corpus. Europe PMC is free; the output includes DOI/PMID, publication date, abstract, PMCID/full-text availability, and a deduplication report.

```bash
python demo_pipeline.py fetch --year-start 2022 --year-end 2023 --limit 1000 --output runs/demo/data/history_raw.jsonl
python demo_pipeline.py fetch --year-start 2024 --year-end 2024 --limit 3000 --output runs/demo/data/validation_2024_raw.jsonl
```

### 2. Extract historical concepts and claims

Extract concepts and claims from the historical abstracts with Qwen3. Inspect a sample of the output before continuing; extraction errors are retained per paper in `extraction_status`.

```bash
python demo_pipeline.py extract --input runs/demo/data/history_raw.jsonl --output runs/demo/data/history_2022_2023.jsonl
```

### 3. Build the historical graph

Build the historical PMI graph and MPNet vectors. Positive-PMI pairs form graph edges, while full pair counts are retained so absent or low co-occurrence remains measurable. The graph also retains extracted claim sentences and their article metadata for relation learning.

```bash
python demo_pipeline.py graph --input runs/demo/data/history_2022_2023.jsonl --output runs/demo/data/history_graph.json
```

### 4. Generate and freeze candidates

Generate up to 500 candidates per method (up to 1,500 method-ranked rows total). `direct_llm` uses Qwen3, `graph_path` uses shared neighbors in the historical graph, and `ours` ranks low-frequency, low-association, semantically distant pairs with a graph-pattern bonus. All methods use the same fixed novelty filters. Candidate JSON and a SHA-256 manifest are written before validation.

```bash
python demo_pipeline.py generate --input runs/demo/data/history_2022_2023.jsonl --graph runs/demo/data/history_graph.json --count 500 --output runs/demo/data/frozen_candidates.json
```

Review the frozen candidate file and manifest before validation. Keep them unchanged for all comparisons.

### 5. Validate against 2024 abstracts

The default mode uses the supplied 3,000-abstract validation JSONL:

```bash
python demo_pipeline.py validate --candidates runs/demo/data/frozen_candidates.json --validation runs/demo/data/validation_2024_raw.jsonl --output runs/demo/data/evidence_2024.json
```

Alternatively, `--targeted-search` queries literature databases separately for each unique relation pair, searching title/abstract fields and retrieving matching abstracts. The default `--targeted-source auto` tries Europe PMC first and falls back to PubMed through NCBI E-utilities if Europe PMC is unavailable; use `--targeted-source pubmed` or `--targeted-source europe_pmc` to select one explicitly. For PubMed, ESearch retrieves matching PMID values and EFetch retrieves abstracts and article metadata. Results are filtered to the requested first publication year (electronic publication date when indexed, otherwise print/publication date), and review articles are excluded. Historical DOI overlaps are excluded.

Targeted search can take time for large candidate sets. `--targeted-max-candidates N` optionally limits it to the first N unique pairs in rank order, interleaved across methods; unqueried pairs are explicitly marked `not_searched`.

```bash
python demo_pipeline.py validate --targeted-search --targeted-max-candidates 50 --candidates runs/demo/data/frozen_candidates.json --output runs/demo/data/evidence_2024_targeted.json
```

Europe PMC and PubMed records report their database source and publication type. Europe PMC targeted searches include preprints and exclude reviews; preprints are labeled as not peer reviewed, and journal-indexed records are not treated as independently verified peer review. Preprint/journal records with the same normalized title are deduplicated so versions cannot count as separate articles. PubMed may not include preprints and only provides free abstracts/metadata; PMC full text is available only for open-access records. PubMed requests are rate-limited to stay within NCBI's unauthenticated E-utilities guidance; setting `NCBI_API_KEY` in the environment permits NCBI's higher rate limit.

Validation uses the existing local Qwen3 checkpoint to label each co-mention sentence. The prompt requires sentence-only evidence and tells the model not to use prior knowledge. A Qwen `support` label is counted directly as model-supported; no human-label step is required. Because Qwen may also generate candidates, these labels are not independent model confirmation and should be interpreted as automated literature triage rather than ground truth. An `ours` relation is retained for learning only when at least three distinct articles support it. The output records the reviewer model path and prompt version; no external model API key is needed for evidence review.

If a broader validation corpus has already been fetched, it can also be used with `--validation` without `--search-report`. Qwen inference runs locally and does not depend on outbound API access.

### 6. Learn reusable relation patterns

Learn relation patterns from qualifying `ours` pairs' historical claim evidence and 2024 model-supported evidence, then generate new concept pairs from the historical vocabulary. Qwen `support` labels count directly, with at least three distinct DOI/PMID/article IDs required per relation. Qwen3 summarizes relation patterns and proposes candidates; MPNet stores a prototype embedding from the relation pattern and evidence context. This is evidence-conditioned prototype learning, not neural relation-encoder training.

The learning command accepts either full frozen-candidate evidence or an `ours`-only evidence subset when its source-candidate hash and exact subset hash both match the frozen manifest. Never use 2025 evidence to learn patterns, adjust thresholds, or change the frozen candidate set.

```bash
python demo_pipeline.py learn --candidates runs/demo/data/frozen_candidates.json --evidence runs/demo/data/evidence_2024.json --graph runs/demo/data/history_graph.json --count 500 --output runs/demo/data/learned_relations.json
```

### 7. Evaluate and audit

Compute per-method raw support rate, Precision@K, Precision@common-K, novelty rate, co-mention rate, and supported relations per 1,000 candidate-document checks:

```bash
python demo_pipeline.py evaluate --candidates runs/demo/data/frozen_candidates.json --evidence runs/demo/data/evidence_2024.json --validation runs/demo/data/validation_2024_raw.jsonl --output-dir runs/demo/results/2024 --k 100
```

Audit failure cases and corpus sizes. Report each method's actual candidate count; use `precision_at_common_k_strict` when methods produce different numbers of candidates. Review extraction failures, alias mismatches, evidence sentences, historical DOI exclusions, and whether a small number of concepts or documents dominate the results.

### Optional 2025 test

Run only after all thresholds and candidate files are fixed. This command reuses the same frozen set and writes separate outputs; do not change candidates based on 2025 results. The 2025 run is evaluation only: never pass its evidence to `learn` or tune candidates and thresholds from its results.

```bash
python demo_pipeline.py fetch --year-start 2025 --year-end 2025 --limit 3000 --output runs/demo/data/test_2025_raw.jsonl
python demo_pipeline.py validate --year 2025 --candidates runs/demo/data/learned_relations.json --validation runs/demo/data/test_2025_raw.jsonl --output runs/demo/data/evidence_2025.json
```

Since the checkpoint may have been pretrained on papers from 2025, this temporal test demonstrates later literature support as classified by the model, not that the model predicted papers it had never seen.

## Reproducibility and Caveats

- `frozen_candidates.manifest.json` records the candidate SHA-256, historical input SHA-256, historical DOI list, thresholds, and temporal boundary. Validation and evaluation refuse a changed candidate list.
- Historical graph construction and generation accept only 2022/2023 documents; validation accepts one requested year (2024 by default, optionally 2025).
- 2024 is the independent evidence corpus for review and learning; 2025 is held out for final prospective evaluation. The learning command rejects evidence later than 2024.
- Qwen is used for both generation and sentence-level evidence classification, so its label is not an independent model verification or ground-truth annotation. Report support metrics as model-labeled literature evidence and consider an independent expert/model audit when interpreting scientific validity.
- `historical_novelty_rate` is an operational threshold metric, not a claim of scientific novelty.
- Europe PMC retrieval covers abstracts, not all full text. Exact string/alias matching can miss paraphrases, and a co-occurrence search can miss a relevant evidence sentence.
- Direct LLM candidates are constrained to the historical concept vocabulary and pass the same novelty filter, making comparisons interpretable but not equivalent to unconstrained generation.
- Model outputs are hypotheses and reviewer labels; inspect the stored evidence sentences and reasons before reporting findings.

## Tests

```bash
python -m unittest -v test_demo_pipeline.py
```
