"""
Predicted-Label Baseline for Oracle Ablation comparison.

Runs SHIELDA recovery with the Exception Classifier (same as the original
experiment) on the same 60 cases, but using the same Claude Haiku model as
run_oracle_ablation.py. This gives a directly-comparable baseline so that
delta = oracle_rate - predicted_rate is model-consistent.
"""

import os, sys, json, csv
from datetime import datetime
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.exception_monitor import ExceptionMonitor
from src.core.recovery_controller import RecoveryController
from src.core.escalator import Escalator
from src.core.agent_simulator import AgentSimulator
from src.core.judge import RecoveryJudge

ORIGINAL_CSV = os.path.join(ROOT, "data/results_master/experiment_stats_1769151076.csv")
GT_DIR       = os.path.join(ROOT, "data/shielda_ground_truth")
LOGS_DIR     = os.path.join(ROOT, "data/logs")
OUT_DIR      = os.path.join(ROOT, "data/results_oracle")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_RETRIES = 3

EQUIVALENCE_GROUPS = {
    "MemoryOverflow": ["ContextWindowExceeded", "HistoryTruncation"],
    "ContextWindowExceeded": ["MemoryOverflow", "HistoryTruncation"],
    "HistoryTruncation": ["MemoryOverflow", "ContextWindowExceeded"],
    "ToolCallFailure": ["APIFailure", "ExternalServiceError"],
    "APIFailure": ["ToolCallFailure", "ExternalServiceError"],
}


def load_original_case_ids():
    cases = []
    with open(ORIGINAL_CSV) as f:
        for row in csv.DictReader(f):
            cases.append({
                "dataset": row["Dataset"],
                "id": row["CaseID"],
                "gt_label": row["GT_Label"],
            })
    return cases


def load_gt_analysis():
    analyses = {}
    for ds in ["gaia", "alfworld", "webshop"]:
        path = os.path.join(GT_DIR, f"{ds}_labels.json")
        with open(path) as f:
            for item in json.load(f):
                sgt = item.get("shielda_gt", {})
                if "error" not in sgt:
                    analyses[item["trajectory_id"]] = sgt.get("analysis", "")
    return analyses


MAX_LOG_CHARS = 8000   # ~6k tokens — same limit as oracle run for consistency

def load_log(dataset, case_id):
    path = os.path.join(LOGS_DIR, dataset, f"{case_id}.json")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        data = json.load(f)
    msgs = data if isinstance(data, list) else data.get("messages", [])
    full = json.dumps(msgs[-20:], indent=2)
    return full[-MAX_LOG_CHARS:]


def run_predicted_recovery(case_id, gt_label, gt_analysis, log_str,
                            monitor, controller, escalator, simulator, judge):
    goal = "Solve the task."
    current_history = log_str
    trace = []

    # ── Phase 1: Classifier → Handler ────────────────────────────────────────
    diag = monitor.diagnose(goal, current_history, "")
    predicted_label = diag.get("exception_name", "Unknown")

    local_exception = predicted_label

    for i in range(MAX_RETRIES):
        pattern = controller.select_pattern(local_exception, "N/A", "Monitor")
        intervention = controller.format_fix_directive(pattern, "Last Error")

        simulated_action = simulator.step(
            history=[{"role": "user", "content": current_history}],
            injected_guidance=intervention,
        )

        is_pass, reason = judge.evaluate(
            goal, current_history, gt_analysis, intervention, simulated_action,
            match_info={"is_match": True, "match_type": "Predicted"},
        )

        trace.append({
            "step": i + 1, "phase": "Local",
            "predicted_label": predicted_label,
            "gt_label": gt_label,
            "correct": (predicted_label == gt_label or
                        gt_label in EQUIVALENCE_GROUPS.get(predicted_label, [])),
            "pattern": pattern.get("name", "Unknown"),
            "judge_pass": is_pass, "reason": reason,
        })

        if is_pass:
            return True, "Local", predicted_label, trace

        current_history += f"\n[System]: {intervention}\n[Agent]: {simulated_action}\n"

    # ── Phase 2: Escalator (unchanged) ────────────────────────────────────────
    esc_res = escalator.analyze_and_fix(goal, local_exception, current_history, "Last Obs", [])
    esc_exception    = esc_res.get("root_cause_exception", "Unknown")
    deep_intervention = esc_res.get("intervention_prompt", "STOP. Re-plan.")

    if esc_exception == "HumanIntervention":
        trace.append({"phase": "Escalator", "outcome": "HumanIntervention"})
        return False, "HumanIntervention", predicted_label, trace

    simulated_action = simulator.step(
        history=[{"role": "user", "content": current_history}],
        injected_guidance=deep_intervention,
    )

    is_pass, reason = judge.evaluate(
        goal, current_history, gt_analysis, deep_intervention, simulated_action,
        match_info={"is_match": True, "match_type": "Escalator/Predicted"},
    )

    trace.append({
        "phase": "Escalator",
        "escalator_diagnosis": esc_exception,
        "judge_pass": is_pass, "reason": reason,
    })

    if is_pass:
        return True, "Escalator", predicted_label, trace

    return False, "Failed", predicted_label, trace


def main():
    print("=== Predicted-Label Baseline (Claude Haiku, same model as Oracle) ===")
    print("Running SHIELDA with Classifier output on original 60 cases.\n")

    monitor    = ExceptionMonitor()
    controller = RecoveryController()
    escalator  = Escalator()
    simulator  = AgentSimulator()
    judge      = RecoveryJudge()

    original_cases = load_original_case_ids()
    analysis_map   = load_gt_analysis()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path  = os.path.join(OUT_DIR, f"predicted_ablation_{timestamp}.csv")
    json_path = os.path.join(OUT_DIR, f"predicted_ablation_{timestamp}_traces.json")

    results = []
    traces  = []

    for case in tqdm(original_cases, desc="Predicted ablation"):
        cid = case["id"]
        ds  = case["dataset"]
        gt_label = case["gt_label"]

        log_str = load_log(ds, cid)
        if not log_str:
            print(f"  ⚠️  No log for {cid}, skipping.")
            continue

        gt_analysis = analysis_map.get(cid, "")

        success, resolved_phase, predicted_label, trace = run_predicted_recovery(
            cid, gt_label, gt_analysis, log_str,
            monitor, controller, escalator, simulator, judge,
        )

        correct = (predicted_label == gt_label or
                   gt_label in EQUIVALENCE_GROUPS.get(predicted_label, []))

        results.append({
            "Dataset":          ds,
            "CaseID":           cid,
            "GT_Label":         gt_label,
            "Predicted_Label":  predicted_label,
            "Correct":          int(correct),
            "Predicted_Success": int(success),
            "ResolvedPhase":    resolved_phase,
        })
        traces.append({"case_id": cid, "dataset": ds,
                        "gt_label": gt_label, "predicted_label": predicted_label,
                        "correct": correct, "success": success, "trace": trace})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Results ===")
    from collections import defaultdict
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r["Dataset"]].append(r)

    total_success = sum(r["Predicted_Success"] for r in results)
    total_correct = sum(r["Correct"] for r in results)
    total = len(results)

    for ds in ["alfworld", "gaia", "webshop"]:
        rows = by_ds[ds]
        s = sum(r["Predicted_Success"] for r in rows)
        c = sum(r["Correct"] for r in rows)
        print(f"  {ds:10s}: recovery={s}/{len(rows)}={s/len(rows):.1%}  acc={c}/{len(rows)}={c/len(rows):.1%}")

    print(f"  {'Overall':10s}: recovery={total_success}/{total}={total_success/total:.1%}  acc={total_correct}/{total}={total_correct/total:.1%}")
    print(f"\n  [Paper]  Full SHIELDA (Gemini, predicted): 66.7%")
    print(f"  [Claude] SHIELDA (Claude Haiku, predicted): {total_success/total:.1%}")
    print(f"  [Claude] Classifier accuracy (Claude Haiku): {total_correct/total:.1%}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as f:
        json.dump(traces, f, indent=2)

    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
