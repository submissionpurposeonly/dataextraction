"""
Oracle Ablation for R1.1 / R2-C6 response.

Runs SHIELDA recovery with Ground-Truth exception labels injected
(bypassing the Exception Classifier) on the same 60 cases used in
the original experiment. Produces a CSV that can be compared
directly to experiment_stats_1769151076.csv.
"""

import os, sys, json, csv
from datetime import datetime
from tqdm import tqdm

# ─── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.recovery_controller import RecoveryController
from src.core.escalator import Escalator
from src.core.agent_simulator import AgentSimulator
from src.core.judge import RecoveryJudge

# ─── config ───────────────────────────────────────────────────────────────────
ORIGINAL_CSV  = os.path.join(ROOT, "data/results_master/experiment_stats_1769151076.csv")
GT_DIR        = os.path.join(ROOT, "data/shielda_ground_truth")
LOGS_DIR      = os.path.join(ROOT, "data/logs")
OUT_DIR       = os.path.join(ROOT, "data/results_oracle")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_RETRIES = 3


def load_original_case_ids():
    """Return the 60 case IDs (and their datasets) from the original experiment."""
    cases = []
    with open(ORIGINAL_CSV) as f:
        for row in csv.DictReader(f):
            cases.append({"dataset": row["Dataset"], "id": row["CaseID"]})
    return cases


def load_gt_labels():
    """Load SHIELDA GT labels keyed by trajectory_id."""
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


MAX_LOG_CHARS = 8000   # ~6k tokens — fits any model; same limit used in predicted run

def load_log(dataset, case_id):
    path = os.path.join(LOGS_DIR, dataset, f"{case_id}.json")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        data = json.load(f)
    msgs = data if isinstance(data, list) else data.get("messages", [])
    full = json.dumps(msgs[-20:], indent=2)
    return full[-MAX_LOG_CHARS:]  # keep most-recent context within token budget


def run_oracle_recovery(case_id, gt_label, gt_analysis, log_str,
                         controller, escalator, simulator, judge):
    """
    Like ExperimentRunnerRQ2._execute_recovery but uses GT label
    instead of calling monitor.diagnose().
    """
    goal = "Solve the task."
    current_history = log_str
    trace = []

    # ── Phase 1: Local Handler with ORACLE label ──────────────────────────────
    local_exception = gt_label          # ← key difference: skip classifier

    for i in range(MAX_RETRIES):
        pattern = controller.select_pattern(local_exception, "N/A", "Monitor")
        intervention = controller.format_fix_directive(pattern, "Last Error")

        simulated_action = simulator.step(
            history=[{"role": "user", "content": current_history}],
            injected_guidance=intervention,
        )

        is_pass, reason = judge.evaluate(
            goal, current_history, gt_analysis, intervention, simulated_action,
            match_info={"is_match": True, "match_type": "Oracle"},
        )

        trace.append({
            "step": i + 1, "phase": "Local",
            "diagnosis": local_exception, "oracle": True,
            "pattern": pattern.get("name", "Unknown"),
            "judge_pass": is_pass, "reason": reason,
        })

        if is_pass:
            return True, "Local", trace

        current_history += f"\n[System]: {intervention}\n[Agent]: {simulated_action}\n"

    # ── Phase 2: Escalator (unchanged) ────────────────────────────────────────
    esc_res = escalator.analyze_and_fix(goal, local_exception, current_history, "Last Obs", [])
    esc_exception    = esc_res.get("root_cause_exception", "Unknown")
    deep_intervention = esc_res.get("intervention_prompt", "STOP. Re-plan.")

    if esc_exception == "HumanIntervention":
        trace.append({"phase": "Escalator", "outcome": "HumanIntervention"})
        return False, "HumanIntervention", trace

    simulated_action = simulator.step(
        history=[{"role": "user", "content": current_history}],
        injected_guidance=deep_intervention,
    )

    is_pass, reason = judge.evaluate(
        goal, current_history, gt_analysis, deep_intervention, simulated_action,
        match_info={"is_match": True, "match_type": "Escalator/Oracle"},
    )

    trace.append({
        "phase": "Escalator",
        "escalator_diagnosis": esc_exception,
        "intervention": deep_intervention,
        "judge_pass": is_pass, "reason": reason,
    })

    if is_pass:
        return True, "Escalator", trace

    return False, "Failed", trace


def main():
    print("=== Oracle Ablation Experiment ===")
    print("Replacing Classifier with Ground-Truth labels; Handler + Escalator unchanged.\n")

    controller = RecoveryController()
    escalator  = Escalator()
    simulator  = AgentSimulator()
    judge      = RecoveryJudge()

    original_cases = load_original_case_ids()   # 60 cases
    gt_map         = load_gt_labels()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path  = os.path.join(OUT_DIR, f"oracle_ablation_{timestamp}.csv")
    json_path = os.path.join(OUT_DIR, f"oracle_ablation_{timestamp}_traces.json")

    results = []
    traces  = []

    for case in tqdm(original_cases, desc="Oracle ablation"):
        cid = case["id"]
        ds  = case["dataset"]

        if cid not in gt_map:
            print(f"  ⚠️  No GT label for {cid}, skipping.")
            continue

        log_str    = load_log(ds, cid)
        if not log_str:
            print(f"  ⚠️  No log for {cid}, skipping.")
            continue

        gt_label   = gt_map[cid]["gt_label"]
        gt_analysis = gt_map[cid]["gt_analysis"]

        success, resolved_phase, trace = run_oracle_recovery(
            cid, gt_label, gt_analysis, log_str,
            controller, escalator, simulator, judge,
        )

        results.append({
            "Dataset":       ds,
            "CaseID":        cid,
            "GT_Label":      gt_label,
            "Oracle_Success": int(success),
            "ResolvedPhase": resolved_phase,
        })
        traces.append({"case_id": cid, "dataset": ds, "gt_label": gt_label,
                        "success": success, "trace": trace})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Results ===")
    from collections import defaultdict
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r["Dataset"]].append(r)

    total_success = sum(r["Oracle_Success"] for r in results)
    total = len(results)

    for ds in ["alfworld", "gaia", "webshop"]:
        rows = by_ds[ds]
        s = sum(r["Oracle_Success"] for r in rows)
        print(f"  {ds:10s}: {s}/{len(rows)} = {s/len(rows):.1%}")

    print(f"  {'Overall':10s}: {total_success}/{total} = {total_success/total:.1%}")
    print(f"\n  [Paper] Full SHIELDA (predicted labels): 66.7%")
    print(f"  [Oracle] SHIELDA + GT labels:            {total_success/total:.1%}")
    delta = total_success/total - 0.667
    print(f"  Delta:                                   {delta:+.1%}")

    # ── Save ──────────────────────────────────────────────────────────────────
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
