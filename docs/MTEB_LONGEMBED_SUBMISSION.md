# MTEB & LongEmbed Submission — Step-by-Step Guide

**Goal:** Get SAID-LAM-v1 on the **MTEB leaderboard** (main scores) and the **LongEmbed / Long-context Retrieval** tab with both scores live.

**Summary:** You need **two PRs** (model wrapper + results). LongEmbed is part of MTEB — the same results PR feeds **both** the main MTEB leaderboard and the Long-context Retrieval (LongEmbed) tab. There is **no separate LongEmbed submission**; your LEMB* task results are what populate LongEmbed.

---

## Prerequisites (must be done first)

- [ ] **HuggingFace:** Model `SAIDResearch/SAID-LAM-v1` is **public** and has weights + config.
- [ ] **PyPI:** `pip install said-lam` works; package is published.
- [ ] **Local check:** `mteb.get_model("SAIDResearch/SAID-LAM-v1")` loads and runs (e.g. on LEMBNeedleRetrieval).

---

## Step 1: Model Wrapper PR (embeddings-benchmark/mteb)

This registers your model so MTEB can load it and run evaluations.

### 1.1 Fork and clone

```bash
gh repo fork embeddings-benchmark/mteb --clone
cd mteb
git checkout -b add-said-lam-v1
```

### 1.2 Add the model file

Create **`mteb/models/said_lam.py`** in the MTEB repo. You can base it on the wrapper in your project. The guide in `docs/COMPLETE_SUBMISSION_GUIDE.md` (Step 4, Part A) has the full code. Key points:

- **Loader:** `said_lam_loader()` that does `from said_lam import LAM` and returns `LAM(model_name, device=device, **kwargs)` (and call `model.auto_activate_mteb()` if needed).
- **Encoder class:** Implements MTEB’s `encode()`, and for retrieval tasks `index()` + `search()` using your engine’s `index_mteb` / `search_mteb`.
- **ModelMeta:** `said_lam_v1` with `name="SAIDResearch/SAID-LAM-v1"`, `embed_dim=384`, `n_parameters=23_848_788`, `languages=["eng"]`, `loader=said_lam_loader`, etc.

Your repo already has `said_lam/said_lam.py` with `ModelMeta` and loader; adapt that into the MTEB repo’s `mteb/models/said_lam.py` so it matches the current MTEB `AbsEncoder` / API (imports from `mteb.models.abs_encoder`, `mteb.models.model_meta`).

### 1.3 Register in `mteb/models/overview.py`

In the MTEB repo, add:

```python
from mteb.models.said_lam import said_lam_v1
```

and ensure `said_lam_v1` is included in the list/dict of models that MTEB exposes (see how other models are registered in that file).

### 1.4 Test locally

```bash
pip install said-lam
python -c "
import mteb
model = mteb.get_model('SAIDResearch/SAID-LAM-v1')
meta = mteb.get_model_meta('SAIDResearch/SAID-LAM-v1')
print(f'Model: {meta.name}, Params: {meta.n_parameters}, Dim: {meta.embed_dim}')
# Optional: run one task
tasks = mteb.get_tasks(tasks=['LEMBNeedleRetrieval'])
evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(model, output_folder='./test_results')
"
```

### 1.5 Open the PR

```bash
git add mteb/models/said_lam.py mteb/models/overview.py
git commit -m "Add SAIDResearch/SAID-LAM-v1 (23.85M param Linear Attention Model)"
gh pr create --title "Add SAIDResearch/SAID-LAM-v1" --body "See description below"
```

In the PR description include:

- Model card link (HuggingFace), `pip install said-lam`, architecture summary.
- Note that it implements both encoding and retrieval (SearchProtocol) for LEMB* tasks.
- Link to HuggingFace: https://huggingface.co/SAIDResearch/SAID-LAM-v1

**Save the PR number** (e.g. `embeddings-benchmark/mteb#XXXX`) for the results PR.

---

## Step 2: Run MTEB evaluations (including LongEmbed tasks)

Run the official MTEB pipeline so the output folder layout matches what the **results** repo expects. Use the **same model name** that will be in the results repo: `SAIDResearch__SAID-LAM-v1` (double underscore).

### 2.1 Tasks to run

- **LongEmbed (LEMB*):** LEMBNeedleRetrieval, LEMBPasskeyRetrieval, LEMBNarrativeQARetrieval, LEMBQMSumRetrieval, LEMBWikiMQARetrieval, LEMBSummScreenFDRetrieval.
- **Classic MTEB (optional but recommended):** e.g. STS12, STS13, STS14, STS15, STS16, STSBenchmark, Banking77Classification, or the full English benchmark.

Example:

```python
import mteb
from said_lam import LAM

model = LAM("SAIDResearch/SAID-LAM-v1")
# Optional: if your wrapper uses engine.auto_activate_mteb(), ensure it's called (e.g. in loader)

tasks = mteb.get_tasks(tasks=[
    "LEMBNeedleRetrieval",
    "LEMBPasskeyRetrieval",
    "LEMBNarrativeQARetrieval",
    "LEMBWikiMQARetrieval",
    "LEMBQMSumRetrieval",
    "LEMBSummScreenFDRetrieval",
    "STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark",
    "Banking77Classification",
], languages=["eng"])

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results/SAIDResearch__SAID-LAM-v1")
```

CLI equivalent (after the model is in MTEB, or using your local fork):

```bash
mteb run -m SAIDResearch/SAID-LAM-v1 -t LEMBNeedleRetrieval LEMBPasskeyRetrieval ... --output-folder results
```

### 2.2 Expected output layout

After `evaluation.run(..., output_folder="results/SAIDResearch__SAID-LAM-v1")` you should see something like:

```
results/
  SAIDResearch__SAID-LAM-v1/
    main/                    # revision
      model_meta.json
      LEMBNeedleRetrieval.json
      LEMBPasskeyRetrieval.json
      ...
```

This folder is what you will submit to **embeddings-benchmark/results**.

---

## Step 3: Results PR (embeddings-benchmark/results)

This is what makes your scores appear on the **MTEB leaderboard** and the **LongEmbed (Long-context Retrieval)** tab.

### 3.1 Fork and clone

```bash
gh repo fork embeddings-benchmark/results --clone
cd results
```

### 3.2 Copy your results

Copy the **entire** model results folder (including revision subfolder) into the results repo. The repo typically has a flat or categorized structure; check the repo’s README or existing folders to see the exact convention (e.g. `SAIDResearch__SAID-LAM-v1/main/` at repo root or under a category).

```bash
cp -r /path/to/results/SAIDResearch__SAID-LAM-v1 .
```

### 3.3 Commit and open PR

```bash
git checkout -b add-said-lam-v1-results
git add SAIDResearch__SAID-LAM-v1/
git commit -m "Add SAIDResearch/SAID-LAM-v1 results"
gh pr create --title "Add SAIDResearch/SAID-LAM-v1 results" --body "Results for SAID-LAM-v1. Model wrapper PR: embeddings-benchmark/mteb#XXXX

Key results:
- LEMBNeedleRetrieval: ...
- LEMBPasskeyRetrieval: ...
- STSBenchmark: ...
"
```

Replace `XXXX` with the actual model wrapper PR number.

---

## Step 4: “Taking it live” — MTEB and LongEmbed scores

- **MTEB leaderboard:** https://huggingface.co/spaces/mteb/leaderboard  
- **LongEmbed:** Same leaderboard, **“Long-context Retrieval”** (or equivalent) tab; it uses the same results repo and shows LEMB* tasks.

There is **no separate “ticket” to take MTEB vs LongEmbed live**. One process covers both:

1. **Model wrapper PR** merged in `embeddings-benchmark/mteb` → model is loadable by MTEB.
2. **Results PR** merged in `embeddings-benchmark/results` → scores are in the data that feeds the leaderboard.

After **both PRs are merged**, the leaderboard (and its LongEmbed tab) is typically updated from the results repo; your model and its scores should appear for both **MTEB scoring** and **LongEmbed**.

### If your model doesn’t appear after merges

- **Open an issue** (or discussion) on the **leaderboard** or **mteb** repo to ask maintainers to refresh or re-run the leaderboard pipeline:
  - Leaderboard Space: https://huggingface.co/spaces/mteb/leaderboard (Discussions/Issues if available).
  - MTEB repo: https://github.com/embeddings-benchmark/mteb/issues  
- In the issue, state clearly:
  - Model: `SAIDResearch/SAID-LAM-v1`
  - Model wrapper PR (merged): link
  - Results PR (merged): link
  - That you expect to see the model on both the **main MTEB leaderboard** and the **LongEmbed / Long-context Retrieval** tab.

That issue serves as your “ticket” to take the scores live if they don’t show up automatically.

---

## Checklist summary

| Step | Action | Repo |
|------|--------|------|
| 1 | Model wrapper PR (model file + registration) | embeddings-benchmark/mteb |
| 2 | Run MTEB evals (LEMB* + optional classic tasks) | Local |
| 3 | Results PR (copy `SAIDResearch__SAID-LAM-v1/` into results) | embeddings-benchmark/results |
| 4 | Both PRs merged → scores live on MTEB + LongEmbed tab | — |
| 5 | If not live: open issue to request leaderboard refresh | mteb/leaderboard or mteb repo |

---

## Quick reference

- **MTEB repo:** https://github.com/embeddings-benchmark/mteb  
- **Results repo:** https://github.com/embeddings-benchmark/results  
- **Leaderboard:** https://huggingface.co/spaces/mteb/leaderboard  
- **Full submission guide (HF, PyPI, GitHub, MTEB):** `docs/COMPLETE_SUBMISSION_GUIDE.md`  
- **Model wrapper code reference:** `said_lam/said_lam.py` (ModelMeta + loader); adapt for `mteb/models/said_lam.py` to match current MTEB API.
