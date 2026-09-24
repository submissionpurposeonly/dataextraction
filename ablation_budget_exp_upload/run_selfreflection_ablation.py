"""
Self-Reflection Baseline Ablation — Reviewer Response Supplement.

Same 60 cases as oracle/predicted ablation. Two configurations:
  (a) uncapped      : max_tries = 4  (matches default in ExperimentRunnerRQ2)
  (b) budget-matched: max_tries = 1  (= mean escalation steps from predicted ablation = 1)

LLM calls and wall-clock timing are instrumented per case.
Results saved to data/results_oracle/ with fresh timestamps (no existing files touched).

NOTE: Independent of paper Table 2 (45.0%) / Table 3 (66.7%).
"""

import os, sys, json, csv, time
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.llm_client import GeminiClient
from src.core.agent_simulator import AgentSimulator
from src.core.judge import RecoveryJudge
from src.core.baselines import BaselineMethods

ORIGINAL_CSV  = os.path.join(ROOT, "data/results_master/experiment_stats_1769151076.csv")
GT_DIR        = os.path.join(ROOT, "data/shielda_ground_truth")
LOGS_DIR      = os.path.join(ROOT, "data/logs")
OUT_DIR       = os.path.join(ROOT, "data/results_oracle")
MAX_LOG_CHARS = 8000

os.makedirs(OUT_DIR, exist_ok=True)


# ── Instrumented LLM client ───────────────────────────────────────────────────

class InstrumentedClient(GeminiClient):
    """Wraps GeminiClient to count calls and measure elapsed time.
    Also fixes the missing `temperature` parameter in generate_text."""

    def __init__(self):
        super().__init__()
        self._call_log: list = []

    # ── fix: accept temperature; record timing ────────────────────────────────
    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        t0 = time.time()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text
        except Exception as e:
            print(f"  ⚠️  generate_text failed: {e}")
            result = ""
        self._call_log.append({"type": "generate_text", "elapsed_s": time.time() - t0})
        return result

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict:
        t0 = time.time()
        result = super().generate_json(system_prompt, user_prompt, temperature)
        self._call_log.append({"type": "generate_json", "elapsed_s": time.time() - t0})
        return result

    def reset_log(self):
        self._call_log.clear()

    def stats(self) -> dict:
        n = len(self._call_log)
        total = round(sum(c["elapsed_s"] for c in self._call_log), 2)
        return {"n_calls": n, "total_elapsed_s": total}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cases():
    cases = []
    with open(ORIGINAL_CSV) as f:
        for row in csv.DictReader(f):
            cases.append({"dataset": row["Dataset"], "id": row["CaseID"]})
    return cases


def load_gt_labels():
    labels = {}
    for ds in ["gaia", "alfworld", "webshop"]:
        path = os.path.join(GT_DIR, f"{ds}_labels.json")
        with open(path) as f:
            for item in json.load(f):
                sgt = item.get("shielda_gt", {})
                if "error" not in sgt:
                    labels[item["trajectory_id"]] = {
                        "gt_label":    sgt.get("layer_2_exception", "Unknown"),
                        "gt_analysis": sgt.get("analysis", ""),
                    }
    return labels


def load_log(dataset, case_id):
    path = os.path.join(LOGS_DIR, dataset, f"{case_id}.json")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        data = json.load(f)
    msgs = data if isinstance(data, list) else data.get("messages", [])
    full = json.dumps(msgs[-20:], indent=2)
    return full[-MAX_LOG_CHARS:]


# ── Self-Reflection recovery loop ─────────────────────────────────────────────

def run_self_reflection_case(case_id, gt_analysis, log_str,
                              baselines, simulator, judge, llm,
                              max_tries: int):
    """
    Pure Self-Reflection recovery loop (no escalation phase).
    Returns (success, resolved_phase, trace, call_stats).
    """
    goal = "Solve the task."
    current_history = log_str
    trace = []
    llm.reset_log()

    for i in range(max_tries):
        error_context = f"Task execution failed (attempt {i + 1})"
        intervention = baselines.get_self_reflection_prompt(goal, current_history, error_context)

        simulated_action = simulator.step(
            history=[{"role": "user", "content": current_history}],
            injected_guidance=intervention,
        )

        is_pass, reason = judge.evaluate(
            goal, current_history, gt_analysis,
            intervention, simulated_action,
            match_info={"is_match": False, "match_type": "SelfReflection"},
        )

        trace.append({
            "step":           i + 1,
            "phase":          "SelfReflection",
            "intervention":   intervention,
            "agent_action":   simulated_action,
            "judge_pass":     is_pass,
            "reason":         reason,
        })

        if is_pass:
            return True, "SelfReflection", trace, llm.stats()

        current_history += (
            f"\n[System]: {intervention}\n[Agent]: {simulated_action}\n"
        )

    return False, "Failed", trace, llm.stats()


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(max_tries: int, label: str):
    print(f"\n{'='*60}")
    print(f"Self-Reflection Ablation: {label}  (max_tries={max_tries})")
    print(f"{'='*60}")

    llm       = InstrumentedClient()
    baselines = BaselineMethods()
    baselines.llm = llm          # replace internal client with instrumented one

    simulator = AgentSimulator()
    simulator.llm = llm          # share instrumented client

    judge     = RecoveryJudge()
    judge.llm = llm              # share instrumented client

    cases  = load_cases()
    gt_map = load_gt_labels()

    results       = []
    traces        = []
    call_stats_all = []
    skipped       = []

    for case in tqdm(cases, desc=label):
        cid = case["id"]
        ds  = case["dataset"]

        if cid not in gt_map:
            skipped.append({"cid": cid, "reason": "no GT label"})
            continue

        log_str = load_log(ds, cid)
        if not log_str:
            skipped.append({"cid": cid, "reason": "no log file"})
            continue

        gt_label    = gt_map[cid]["gt_label"]
        gt_analysis = gt_map[cid]["gt_analysis"]

        try:
            success, resolved_phase, trace, call_stats = run_self_reflection_case(
                cid, gt_analysis, log_str,
                baselines, simulator, judge, llm,
                max_tries=max_tries,
            )
        except Exception as e:
            print(f"  ⚠️  Case {cid} crashed: {e}")
            skipped.append({"cid": cid, "reason": str(e)})
            continue

        results.append({
            "Dataset":       ds,
            "CaseID":        cid,
            "GT_Label":      gt_label,
            "SR_Success":    int(success),
            "ResolvedPhase": resolved_phase,
            "n_llm_calls":   call_stats["n_calls"],
            "elapsed_s":     call_stats["total_elapsed_s"],
        })
        call_stats_all.append(call_stats)
        traces.append({
            "case_id":    cid,
            "dataset":    ds,
            "gt_label":   gt_label,
            "success":    success,
            "call_stats": call_stats,
            "trace":      trace,
        })

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    safe_label = label.replace(" ", "_")
    csv_path   = os.path.join(OUT_DIR, f"sr_ablation_{safe_label}_{timestamp}.csv")
    json_path  = os.path.join(OUT_DIR, f"sr_ablation_{safe_label}_{timestamp}_traces.json")

    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    with open(json_path, "w") as f:
        json.dump(traces, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r["Dataset"]].append(r)

    total_success = sum(r["SR_Success"] for r in results)
    total         = len(results)

    print(f"\n  Results ({label}):")
    for ds in ["alfworld", "gaia", "webshop"]:
        rows = by_ds[ds]
        s    = sum(r["SR_Success"] for r in rows)
        pct  = f"{s/len(rows)*100:.1f}%" if rows else "N/A"
        print(f"    {ds:10s}: {s:2d}/{len(rows):2d}  ({pct})")

    if total:
        print(f"    {'Overall':10s}: {total_success:2d}/{total:2d}  ({total_success/total*100:.1f}%)")

    if call_stats_all:
        avg_calls   = sum(c["n_calls"]       for c in call_stats_all) / len(call_stats_all)
        avg_elapsed = sum(c["total_elapsed_s"] for c in call_stats_all) / len(call_stats_all)
        total_calls = sum(c["n_calls"]       for c in call_stats_all)
        print(f"\n  LLM calls: {total_calls} total  |  avg {avg_calls:.1f}/case  |  avg {avg_elapsed:.1f}s/case")

    if skipped:
        print(f"  ⚠️  Skipped {len(skipped)}: {[s['cid'] for s in skipped]}")

    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {json_path}")

    return {
        "label":         label,
        "max_tries":     max_tries,
        "results":       results,
        "total_success": total_success,
        "total":         total,
        "call_stats":    call_stats_all,
        "by_ds":         by_ds,
        "skipped":       skipped,
        "csv_path":      csv_path,
        "json_path":     json_path,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Self-Reflection Ablation for Reviewer Response")
    print("Budget cap derivation: ALL 34 escalated cases in predicted ablation used")
    print("exactly 1 escalation step → budget-matched max_tries = 1.\n")

    result_a = run_experiment(max_tries=4, label="uncapped_max4")
    result_b = run_experiment(max_tries=1, label="budget_matched_max1")

    # ── Final comparison ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    header = f"{'Condition':<28} {'ALF':>6} {'GAIA':>6} {'WEB':>6} {'Overall':>10} {'Calls/case':>12} {'s/case':>8}"
    print(header)
    print("-"*60)

    for r in [result_a, result_b]:
        by_ds = r["by_ds"]
        cols  = []
        for ds in ["alfworld", "gaia", "webshop"]:
            rows = by_ds[ds]
            s    = sum(x["SR_Success"] for x in rows)
            cols.append(f"{s}/{len(rows)}")

        ov = f"{r['total_success']}/{r['total']} ({r['total_success']/r['total']*100:.1f}%)" if r["total"] else "N/A"
        cs = r["call_stats"]
        avg_c = f"{sum(c['n_calls'] for c in cs)/len(cs):.1f}" if cs else "N/A"
        avg_t = f"{sum(c['total_elapsed_s'] for c in cs)/len(cs):.1f}" if cs else "N/A"
        print(f"  {r['label']:<26} {cols[0]:>6} {cols[1]:>6} {cols[2]:>6} {ov:>18} {avg_c:>8} {avg_t:>8}")

    print("="*60)
    print("\nNOTE: These results are from a self-contained 60-case (20 ALF / 20 GAIA / 20 WEB)")
    print("supplementary study using Claude Haiku 4.5. They are INDEPENDENT of paper")
    print("Table 2 (45.0% Diag Acc) and Table 3 (66.7% Recovery) which used 200 cases")
    print("with Gemini 2.0 Flash and cannot be directly compared.")
