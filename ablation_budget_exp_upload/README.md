# SHIELDA — Supplementary Experiments for Reviewer Response

This repository contains scripts and raw results for the supplementary ablation
study prepared in response to reviewer comments (R1.1 / R2.6).  
All experiments use **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) as
the underlying model (Gemini 2.0 Flash, used in the original paper, was
deprecated before this study was conducted).  
These results are **independent** of the main paper experiments (Table 2 / Table 3),
which were run on 200 cases with Gemini 2.0 Flash.

---

## Experiment Overview

### Experiment 1 — Oracle Ablation
**Script:** `run_oracle_ablation.py`  
**What it does:** Bypasses the Exception Classifier entirely; injects the
ground-truth exception label directly into the Handler Pattern Registry.
The Handler and Escalation Controller run unchanged.  
This isolates the contribution of the Classifier from the rest of the pipeline.  
**Dataset:** 60 cases (20 ALFWorld / 20 GAIA / 20 WebShop), same case IDs
as the main paper experiment.  
**Outputs → `01_oracle_ablation/`:**
- `oracle_ablation_20260911_1404.csv` — per-case result (final run)
- `oracle_ablation_20260911_1404_traces.json` — per-case intervention traces (final run)
- `oracle_ablation_raw.json` — computed aggregates across oracle and predicted conditions
- `oracle_ablation_20260911_1233.*`, `oracle_ablation_20260911_1304.*` — earlier intermediate runs (kept for completeness)

---

### Experiment 2 — Predicted Ablation
**Script:** `run_predicted_ablation.py`  
**What it does:** Runs the full SHIELDA pipeline on the same 60 cases using
the actual ExceptionMonitor classifier for first-pass diagnosis (no oracle injection).  
Records per-case: GT label, predicted label, first-pass classification correctness,
recovery success, resolved phase (Local / Escalator / Failed).  
**Outputs → `02_predicted_ablation/`:**
- `predicted_ablation_20260911_1404.csv` — per-case result (final run)
- `predicted_ablation_20260911_1404_traces.json` — per-case traces (final run)
- `predicted_ablation_20260911_1305.*` — earlier intermediate run

---

### Experiment 3 — Self-Reflection Baseline Ablation
**Script:** `run_selfreflection_ablation.py`  
**What it does:** Runs a Self-Reflection baseline (critic LLM generates a
general reflective question; no specific exception diagnosis or handler patterns)
on the same 60 cases.  
Two configurations:
- **Uncapped** (`max_tries=4`): up to 4 complete reflection–simulate–judge iterations
- **Budget-matched** (`max_tries=1`): 1 iteration (see note below on budget definition)

**Note on budget-matching:** The Escalation Controller in the predicted ablation
always uses exactly 1 escalation step (after 3 failed local retries).
`max_tries=1` matches the number of escalation-phase interventions, not the total
SHIELDA intervention count (3 local + 1 escalation = 4 total). See the reviewer
response document for the full discussion.  
**Outputs → `03_selfreflection_ablation/`:**
- `sr_ablation_uncapped_max4_20260922_1811.csv` / `_traces.json`
- `sr_ablation_budget_matched_max1_20260922_1936.csv` / `_traces.json`

---

### Experiment 4 — Token-Instrumented Four-Condition Re-run
**Script:** `run_token_instrumented.py`  
**What it does:** Re-runs all four conditions (oracle, predicted, SR-uncapped,
SR-budget-matched) with LLM call instrumentation that captures
`input_tokens` and `output_tokens` from `response.usage` for every API call.
Timing (wall-clock seconds per call) is also recorded.  
**Note:** Recovery rates in this re-run may differ slightly from Experiments 1–3
due to LLM non-determinism (Escalator uses `temperature=0.1`;
SR critic uses `temperature=0.7`; all other components use `temperature=0.0`).  
**Outputs → `04_token_instrumented/`:**
- `token_oracle_20260923_1603.csv` — per-case token stats, oracle condition
- `token_predicted_20260923_1603.csv` — per-case token stats, predicted condition
- `token_sr_uncapped_20260923_1603.csv` — per-case token stats, SR uncapped
- `token_sr_budget_20260923_1603.csv` — per-case token stats, SR budget-matched
- `token_summary_20260923_1603.json` — condition-level averages (calls/case, input tok/case, output tok/case, s/case)

---

### Supporting Script — Ground Truth Refinement
**Script:** `refine_ground_truth.py`  
**What it does:** Converts legacy error labels (coarse module-level categories
such as "plan", "memory", "action") into SHIELDA's 36-type Layer-2 taxonomy
using LLM-assisted mapping. Generates the ground-truth label files consumed
by all ablation scripts.  
This script was run once to produce the GT labels; it is included here for
reproducibility.

---

## Numbers Reported in the Reviewer Response Letter (R2.6)

The paragraph in the reviewer response letter reads:

> *"On this sample, SHIELDA recovered 76.7% (46/60), averaging 9.5 LLM calls, 22,132 input
> tokens, 2,610 output tokens, and 36.9 seconds per case. … at 4 attempts it recovered 31.7%
> (19/60), averaging 7.5 LLM calls, 11,454 input tokens, 1,812 output tokens, and 26.6 seconds
> per case; at 1 attempt it recovered 21.7% (13/60), averaging 3.0 LLM calls, 4,266 input
> tokens, 734 output tokens, and 10.7 seconds per case."*

Each number is traceable to a specific file in this repository.

### Recovery rates — from Experiments 2 and 3 (primary ablation runs)

| Claim in letter | Value | Source file |
|---|---|---|
| SHIELDA (predicted labels) | 76.7% (46/60) | `02_predicted_ablation/predicted_ablation_20260911_1404.csv` |
| Self-Reflection, 4 attempts | 31.7% (19/60) | `03_selfreflection_ablation/sr_ablation_uncapped_max4_20260922_1811.csv` |
| Self-Reflection, 1 attempt | 21.7% (13/60) | `03_selfreflection_ablation/sr_ablation_budget_matched_max1_20260922_1936.csv` |
| Oracle SHIELDA | 88.3% (53/60) | `01_oracle_ablation/oracle_ablation_20260911_1404.csv` |

### LLM calls, token counts, and wall-clock timing — from Experiment 4 (token-instrumented re-run)

All call counts, token counts, and timing figures come from
`04_token_instrumented/token_summary_20260923_1603.json`:

| Condition key in JSON | Calls/case | Input tok/case | Output tok/case | s/case |
|---|---|---|---|---|
| `"predicted"` (SHIELDA) | 9.5 | 22,132 | 2,610 | 36.9 |
| `"sr_uncapped"` (4 attempts) | 7.5 | 11,454 | 1,812 | 26.6 |
| `"sr_budget"` (1 attempt) | 3.0 | 4,266 | 734 | 10.7 |

**Why recovery rates in `token_summary` differ from the letter:**
Experiment 4 was conducted solely to capture token and timing statistics not instrumented
in the original runs. Because the Escalation Controller uses `temperature=0.1` and the SR
critic uses `temperature=0.7`, recovery rates vary slightly across runs due to LLM
non-determinism (Experiment 4 shows 66.7% and 30.0% for SHIELDA predicted and SR uncapped,
respectively). The letter uses recovery rates from Experiments 2–3 as the definitive ablation
results; Experiment 4 supplies only the cost statistics.

---

## Usage

Scripts are designed to run from the repository root (`shielda-exp/`) with Python 3.9+.  
Dependencies: `anthropic`, `python-dotenv`, `tqdm`.  
API key: set `ANTHROPIC_API_KEY` in a `.env` file at the repo root (not included).

```bash
cd shielda-exp/
python ablation_budget_exp/run_oracle_ablation.py
python ablation_budget_exp/run_predicted_ablation.py
python ablation_budget_exp/run_selfreflection_ablation.py
python ablation_budget_exp/run_token_instrumented.py
```
