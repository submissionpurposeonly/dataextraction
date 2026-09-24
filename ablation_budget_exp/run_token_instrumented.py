"""
Token-instrumented re-run of all four ablation conditions.

Captures input_tokens + output_tokens from response.usage for every API call.
Saves NEW timestamped files — does NOT touch existing results_oracle/ files.

Conditions:
  1. oracle          — GT labels injected, Handler + Escalator unchanged
  2. predicted       — ExceptionMonitor classifier + Handler + Escalator
  3. sr_uncapped     — Self-Reflection, max_tries=3
  4. sr_budget       — Self-Reflection, max_tries=1 (budget-matched)
"""

import os, sys, json, csv, time
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.llm_client import GeminiClient
from src.core.recovery_controller import RecoveryController
from src.core.escalator import Escalator
from src.core.agent_simulator import AgentSimulator
from src.core.judge import RecoveryJudge
from src.core.exception_monitor import ExceptionMonitor
from src.core.baselines import BaselineMethods

ORIGINAL_CSV  = os.path.join(ROOT, "data/results_master/experiment_stats_1769151076.csv")
GT_DIR        = os.path.join(ROOT, "data/shielda_ground_truth")
LOGS_DIR      = os.path.join(ROOT, "data/logs")
OUT_DIR       = os.path.join(ROOT, "data/results_oracle")
MAX_LOG_CHARS = 8000
MAX_RETRIES   = 3

EQUIVALENCE_GROUPS = {
    "Hallucinated Facts": ["Observation Hallucination", "Plan Hallucination", "Faulty Attribution"],
    "Circular or Invalid Reasoning": ["Circular Reasoning", "Logic Error"],
    "Overextended Planning": ["Inefficient Plan", "Step Limit"],
    "Tool Invocation Exception": ["Tool Failure", "Tool Error"],
}

os.makedirs(OUT_DIR, exist_ok=True)


# ── Token-instrumented LLM client ─────────────────────────────────────────────

class TokenInstrumentedClient(GeminiClient):
    """
    Wraps GeminiClient. Records per-call:
      - elapsed wall-clock seconds
      - input_tokens  (from response.usage)
      - output_tokens (from response.usage)
    Also fixes the missing `temperature` param in generate_text.
    """

    def __init__(self):
        super().__init__()
        self._call_log: list = []

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        t0 = time.time()
        input_tok = output_tok = 0
        result = ""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text
            if hasattr(response, "usage"):
                input_tok  = response.usage.input_tokens
                output_tok = response.usage.output_tokens
        except Exception as e:
            print(f"  ⚠️  generate_text failed: {e}")
        self._call_log.append({
            "type": "generate_text",
            "elapsed_s": round(time.time() - t0, 2),
            "input_tokens": input_tok,
            "output_tokens": output_tok,
        })
        return result

    def generate_json(self, system_prompt: str, user_prompt: str,
                      temperature: float = 0.0) -> dict:
        t0 = time.time()
        input_tok = output_tok = 0
        result = {"error": "API_FAILED"}
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                if hasattr(response, "usage"):
                    input_tok  = response.usage.input_tokens
                    output_tok = response.usage.output_tokens
                text = response.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                result = json.loads(text)
                break
            except json.JSONDecodeError:
                try:
                    start, end = text.index("{"), text.rindex("}") + 1
                    result = json.loads(text[start:end])
                    break
                except Exception:
                    pass
                print(f"  ⚠️  JSON parse failed (attempt {attempt+1}/3)")
            except Exception as e:
                print(f"  ⚠️  API call failed (attempt {attempt+1}/3): {e}")
                time.sleep(2)
        elapsed = round(time.time() - t0, 2)
        self._call_log.append({
            "type": "generate_json",
            "elapsed_s": elapsed,
            "input_tokens": input_tok,
            "output_tokens": output_tok,
        })
        return result

    def reset_log(self):
        self._call_log.clear()

    def stats(self) -> dict:
        n  = len(self._call_log)
        ts = round(sum(c["elapsed_s"]     for c in self._call_log), 2)
        ti = sum(c["input_tokens"]        for c in self._call_log)
        to = sum(c["output_tokens"]       for c in self._call_log)
        return {"n_calls": n, "elapsed_s": ts,
                "input_tokens": ti, "output_tokens": to}


# ── Data loaders (shared) ──────────────────────────────────────────────────────

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


def _diag_correct(predicted, gt):
    return (predicted == gt or
            gt in EQUIVALENCE_GROUPS.get(predicted, []))


# ── Per-case recovery functions ────────────────────────────────────────────────

def run_oracle_case(gt_label, gt_analysis, log_str,
                     controller, escalator, simulator, judge, llm):
    goal = "Solve the task."
    history = log_str
    trace = []
    llm.reset_log()

    for i in range(MAX_RETRIES):
        pattern = controller.select_pattern(gt_label, "N/A", "Monitor")
        intervention = controller.format_fix_directive(pattern, "Last Error")
        action = simulator.step(
            history=[{"role": "user", "content": history}],
            injected_guidance=intervention,
        )
        passed, reason = judge.evaluate(
            goal, history, gt_analysis, intervention, action,
            match_info={"is_match": True, "match_type": "Oracle"},
        )
        trace.append({"step": i+1, "phase": "Local", "judge_pass": passed})
        if passed:
            return True, "Local", trace, llm.stats()
        history += f"\n[System]: {intervention}\n[Agent]: {action}\n"

    esc = escalator.analyze_and_fix(goal, gt_label, history, "Last Obs", [])
    if esc.get("root_cause_exception") == "HumanIntervention":
        return False, "HumanIntervention", trace, llm.stats()
    deep_intervention = esc.get("intervention_prompt", "STOP. Re-plan.")
    action = simulator.step(
        history=[{"role": "user", "content": history}],
        injected_guidance=deep_intervention,
    )
    passed, reason = judge.evaluate(
        goal, history, gt_analysis, deep_intervention, action,
        match_info={"is_match": True, "match_type": "Escalator/Oracle"},
    )
    trace.append({"step": "esc", "phase": "Escalator", "judge_pass": passed})
    return passed, ("Escalator" if passed else "Failed"), trace, llm.stats()


def run_predicted_case(gt_label, gt_analysis, log_str,
                        monitor, controller, escalator, simulator, judge, llm):
    goal = "Solve the task."
    history = log_str
    trace = []
    llm.reset_log()

    diag = monitor.diagnose(goal, history, "Last observation")
    predicted_label = diag.get("exception_name", "Unknown")
    correct = _diag_correct(predicted_label, gt_label)

    for i in range(MAX_RETRIES):
        pattern = controller.select_pattern(predicted_label, "N/A", "Monitor")
        intervention = controller.format_fix_directive(pattern, "Last Error")
        action = simulator.step(
            history=[{"role": "user", "content": history}],
            injected_guidance=intervention,
        )
        passed, reason = judge.evaluate(
            goal, history, gt_analysis, intervention, action,
            match_info={"is_match": correct, "match_type": ("Exact" if correct else "None")},
        )
        trace.append({"step": i+1, "phase": "Local", "judge_pass": passed})
        if passed:
            return True, "Local", correct, predicted_label, trace, llm.stats()
        history += f"\n[System]: {intervention}\n[Agent]: {action}\n"

    esc = escalator.analyze_and_fix(goal, predicted_label, history, "Last Obs", [])
    if esc.get("root_cause_exception") == "HumanIntervention":
        return False, "HumanIntervention", correct, predicted_label, trace, llm.stats()
    deep_intervention = esc.get("intervention_prompt", "STOP. Re-plan.")
    action = simulator.step(
        history=[{"role": "user", "content": history}],
        injected_guidance=deep_intervention,
    )
    passed, reason = judge.evaluate(
        goal, history, gt_analysis, deep_intervention, action,
        match_info={"is_match": True, "match_type": "Escalator"},
    )
    trace.append({"step": "esc", "phase": "Escalator", "judge_pass": passed})
    return passed, ("Escalator" if passed else "Failed"), correct, predicted_label, trace, llm.stats()


def run_sr_case(gt_analysis, log_str,
                 baselines, simulator, judge, llm, max_tries):
    goal = "Solve the task."
    history = log_str
    trace = []
    llm.reset_log()

    for i in range(max_tries):
        intervention = baselines.get_self_reflection_prompt(
            goal, history, f"Task execution failed (attempt {i+1})")
        action = simulator.step(
            history=[{"role": "user", "content": history}],
            injected_guidance=intervention,
        )
        passed, reason = judge.evaluate(
            goal, history, gt_analysis, intervention, action,
            match_info={"is_match": False, "match_type": "SelfReflection"},
        )
        trace.append({"step": i+1, "phase": "SelfReflection", "judge_pass": passed})
        if passed:
            return True, "SelfReflection", trace, llm.stats()
        history += f"\n[System]: {intervention}\n[Agent]: {action}\n"

    return False, "Failed", trace, llm.stats()


# ── Condition runners ──────────────────────────────────────────────────────────

def run_condition(condition_name, cases, gt_map, run_fn, components):
    print(f"\n{'='*60}")
    print(f"Condition: {condition_name}")
    print(f"{'='*60}")

    results, token_stats, skipped = [], [], []

    for case in tqdm(cases, desc=condition_name):
        cid, ds = case["id"], case["dataset"]
        if cid not in gt_map:
            skipped.append(cid); continue
        log_str = load_log(ds, cid)
        if not log_str:
            skipped.append(cid); continue

        gt = gt_map[cid]
        try:
            row, stats = run_fn(gt, log_str, cid, ds, components)
        except Exception as e:
            print(f"  ⚠️  {cid}: {e}")
            skipped.append(cid); continue

        results.append(row)
        token_stats.append(stats)

    # summary
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r["Dataset"]].append(r)
    total_s = sum(r["Success"] for r in results)
    total   = len(results)

    print(f"\n  Results:")
    for ds in ["alfworld", "gaia", "webshop"]:
        rows = by_ds[ds]
        s = sum(r["Success"] for r in rows)
        print(f"    {ds:10s}: {s}/{len(rows)}")
    if total:
        print(f"    {'Overall':10s}: {total_s}/{total} = {total_s/total*100:.1f}%")

    if token_stats:
        avg = lambda key: round(sum(c[key] for c in token_stats) / len(token_stats), 1)
        print(f"  Avg/case: calls={avg('n_calls')}, "
              f"in_tok={avg('input_tokens')}, out_tok={avg('output_tokens')}, "
              f"elapsed={avg('elapsed_s')}s")

    if skipped:
        print(f"  Skipped {len(skipped)}: {skipped}")

    return results, token_stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    cases  = load_cases()
    gt_map = load_gt_labels()

    all_results = {}  # condition → (results_list, token_stats_list)

    # ── 1. Oracle ──────────────────────────────────────────────────────────────
    llm = TokenInstrumentedClient()
    controller = RecoveryController(); controller.llm = llm
    escalator  = Escalator();          escalator.llm  = llm
    simulator  = AgentSimulator();     simulator.llm  = llm
    judge      = RecoveryJudge();      judge.llm      = llm

    def oracle_fn(gt, log_str, cid, ds, _):
        success, phase, trace, stats = run_oracle_case(
            gt["gt_label"], gt["gt_analysis"], log_str,
            controller, escalator, simulator, judge, llm)
        return ({"Dataset": ds, "CaseID": cid, "GT_Label": gt["gt_label"],
                 "Success": int(success), "ResolvedPhase": phase}, stats)

    r, ts = run_condition("oracle", cases, gt_map, oracle_fn, None)
    all_results["oracle"] = (r, ts)
    _save(r, ts, "oracle", timestamp)

    # ── 2. Predicted ───────────────────────────────────────────────────────────
    llm = TokenInstrumentedClient()
    monitor    = ExceptionMonitor();   monitor.llm    = llm
    controller = RecoveryController(); controller.llm = llm
    escalator  = Escalator();          escalator.llm  = llm
    simulator  = AgentSimulator();     simulator.llm  = llm
    judge      = RecoveryJudge();      judge.llm      = llm

    def predicted_fn(gt, log_str, cid, ds, _):
        success, phase, correct, pred_label, trace, stats = run_predicted_case(
            gt["gt_label"], gt["gt_analysis"], log_str,
            monitor, controller, escalator, simulator, judge, llm)
        return ({"Dataset": ds, "CaseID": cid, "GT_Label": gt["gt_label"],
                 "Predicted_Label": pred_label, "Correct": int(correct),
                 "Success": int(success), "ResolvedPhase": phase}, stats)

    r, ts = run_condition("predicted", cases, gt_map, predicted_fn, None)
    all_results["predicted"] = (r, ts)
    _save(r, ts, "predicted", timestamp)

    # ── 3. SR uncapped (max_tries=3) ───────────────────────────────────────────
    llm = TokenInstrumentedClient()
    baselines  = BaselineMethods();    baselines.llm  = llm
    simulator  = AgentSimulator();     simulator.llm  = llm
    judge      = RecoveryJudge();      judge.llm      = llm

    def sr_uncapped_fn(gt, log_str, cid, ds, _):
        success, phase, trace, stats = run_sr_case(
            gt["gt_analysis"], log_str, baselines, simulator, judge, llm, max_tries=3)
        return ({"Dataset": ds, "CaseID": cid, "GT_Label": gt["gt_label"],
                 "Success": int(success), "ResolvedPhase": phase}, stats)

    r, ts = run_condition("sr_uncapped", cases, gt_map, sr_uncapped_fn, None)
    all_results["sr_uncapped"] = (r, ts)
    _save(r, ts, "sr_uncapped", timestamp)

    # ── 4. SR budget-matched (max_tries=1) ────────────────────────────────────
    llm = TokenInstrumentedClient()
    baselines  = BaselineMethods();    baselines.llm  = llm
    simulator  = AgentSimulator();     simulator.llm  = llm
    judge      = RecoveryJudge();      judge.llm      = llm

    def sr_budget_fn(gt, log_str, cid, ds, _):
        success, phase, trace, stats = run_sr_case(
            gt["gt_analysis"], log_str, baselines, simulator, judge, llm, max_tries=1)
        return ({"Dataset": ds, "CaseID": cid, "GT_Label": gt["gt_label"],
                 "Success": int(success), "ResolvedPhase": phase}, stats)

    r, ts = run_condition("sr_budget", cases, gt_map, sr_budget_fn, None)
    all_results["sr_budget"] = (r, ts)
    _save(r, ts, "sr_budget", timestamp)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("TOKEN INSTRUMENTATION SUMMARY")
    print("="*72)
    hdr = f"{'Condition':<20} {'N':>4} {'Rec%':>7} {'Calls':>7} {'InTok':>8} {'OutTok':>8} {'s/case':>7}"
    print(hdr)
    print("-"*72)

    summary_rows = []
    for cname in ["oracle", "predicted", "sr_uncapped", "sr_budget"]:
        res, toks = all_results[cname]
        n = len(res)
        if n == 0:
            continue
        rec = sum(r["Success"] for r in res) / n * 100
        avg = lambda k: round(sum(c[k] for c in toks) / len(toks), 1) if toks else 0
        row = {
            "condition":     cname,
            "n_cases":       n,
            "recovery_pct":  round(rec, 1),
            "avg_calls":     avg("n_calls"),
            "avg_input_tok": avg("input_tokens"),
            "avg_output_tok":avg("output_tokens"),
            "avg_elapsed_s": avg("elapsed_s"),
        }
        summary_rows.append(row)
        print(f"  {cname:<20} {n:>4} {rec:>6.1f}% {row['avg_calls']:>7.1f} "
              f"{row['avg_input_tok']:>8.0f} {row['avg_output_tok']:>8.0f} {row['avg_elapsed_s']:>7.1f}")

    print("="*72)

    # save summary JSON
    summary_path = os.path.join(OUT_DIR, f"token_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # update reviewer_response_results.md
    _append_token_section(summary_rows, timestamp)


def _save(results, token_stats, label, timestamp):
    if not results:
        return
    csv_path = os.path.join(OUT_DIR, f"token_{label}_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) +
                                ["n_calls", "input_tokens", "output_tokens", "elapsed_s"])
        writer.writeheader()
        for row, stats in zip(results, token_stats):
            writer.writerow({**row, **stats})
    print(f"  Saved: {csv_path}")


def _append_token_section(summary_rows, timestamp):
    md_path = os.path.join(ROOT, "reviewer_response_results.md")
    if not os.path.exists(md_path):
        return

    lines = [
        "\n---\n",
        "## Token Usage — All Four Conditions (Re-run with Instrumentation)\n\n",
        f"**Re-run date:** {timestamp[:8][:4]}-{timestamp[:8][4:6]}-{timestamp[:8][6:]}  \n",
        "**Model:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)  \n",
        "**Source:** `token_summary_" + timestamp + ".json`  \n\n",
        "| Condition | N | Recovery | Avg calls/case | Avg input tok/case | Avg output tok/case | Avg s/case |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    labels = {
        "oracle":      "Oracle SHIELDA (GT labels)",
        "predicted":   "Predicted SHIELDA (classifier)",
        "sr_uncapped": "Self-Reflection uncapped (max_tries=3)",
        "sr_budget":   "Self-Reflection budget-matched (max_tries=1)",
    }
    for r in summary_rows:
        n_s = sum(1 for _ in [r])  # just for label lookup
        rec = f"{r['recovery_pct']}%"
        lines.append(
            f"| {labels.get(r['condition'], r['condition'])} "
            f"| {r['n_cases']} | {rec} "
            f"| {r['avg_calls']} | {r['avg_input_tok']:.0f} "
            f"| {r['avg_output_tok']:.0f} | {r['avg_elapsed_s']} |\n"
        )
    lines.append(
        "\n**Note:** Recovery rates above may differ slightly from the original "
        "2026-09-11 runs due to LLM non-determinism (temperature=0 for all judge/classifier "
        "calls, temperature=0.1 for Escalator, temperature=0.7 for SR critic).\n"
    )

    with open(md_path, "a") as f:
        f.writelines(lines)
    print(f"  Appended token section to {md_path}")


if __name__ == "__main__":
    main()
