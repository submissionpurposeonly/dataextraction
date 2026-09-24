import os
import sys
import json
import difflib
from tqdm import tqdm

# ================= 🚑 路径适配 =================
current_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_path)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)
# ===================================================

from src.utils.llm_client import GeminiClient

class GroundTruthRefiner:
    def __init__(self):
        print("🔧 Initializing Ground Truth Refiner (Robust Mode)...")
        self.llm = GeminiClient()
        self.data_root = os.path.join(project_root, "data")
        self.old_gt_dir = os.path.join(self.data_root, "ground_truth")
        self.logs_dir = os.path.join(self.data_root, "logs")
        self.output_dir = os.path.join(self.data_root, "shielda_ground_truth")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载 Taxonomy
        self.taxonomy_db = self._load_taxonomy_db()
        self.valid_l2_keys = list(self.taxonomy_db.keys())

        # Legacy Definitions
        self.LEGACY_DEFINITIONS = {
            "plan": "Failures in generating, prioritizing, or structuring the sequence of steps.",
            "planning": "Failures in generating, prioritizing, or structuring the sequence of steps.",
            "memory": "Failures in storing, retrieving, or maintaining context.",
            "reflection": "Failures in evaluating self-progress or interpreting outcomes.",
            "action": "Failures in the execution phase (tools, API calls).",
            "system": "Failures originating from the environment, infrastructure, or limits (steps/tokens)."
        }

        # Bridge Map
        self.BRIDGE_MAP = {
            "plan": ["Planning", "Reasoning", "Goal", "Context"],
            "planning": ["Planning", "Reasoning", "Goal", "Context"],
            "memory": ["Memory", "Knowledge Base"],
            "reflection": ["Reflection", "Reasoning", "Task Flow"], 
            "action": ["Tool", "Interface", "Model", "External System", "Action"], 
            "system": ["System", "External System", "Tool", "Interface", "Model"] # Added Model for Token Limit
        }

        # 🔥 硬编码映射：解决常见 Unknown 问题
        self.HARD_MAPPINGS = {
            "step_limit": "Step Limit Exhaustion",
            "step limit": "Step Limit Exhaustion",
            "token_limit": "Token Limit Exceeded",
            "json_error": "Output Validation Failure",
            "hallucination": "Hallucinated Facts"
        }

    def _load_taxonomy_db(self):
        candidates = [
            os.path.join(self.data_root, "taxonomy.json"),
            os.path.join(project_root, "taxonomy.json"),
            os.path.join(project_root, "config", "taxonomy.json")
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    with open(p, 'r') as f: return json.load(f)
                except: pass
        return {}

    def _get_log_slice(self, dataset, case_id, critical_step):
        # ... (Log Reading Logic - same as before) ...
        # For brevity, using simplified version. Ensure you use the full version if available.
        candidates = [
            os.path.join(self.logs_dir, dataset, f"{case_id}.json"),
            os.path.join(self.logs_dir, dataset, f"task_{case_id}.json"),
            os.path.join(self.logs_dir, dataset, f"{case_id.split('_')[0]}.json")
        ]
        log_path = None
        for p in candidates:
            if os.path.exists(p): log_path = p; break
        
        if not log_path:
            folder = os.path.join(self.logs_dir, dataset)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if case_id in f and f.endswith(".json") and "labels" not in f:
                        log_path = os.path.join(folder, f); break
        if not log_path: return "Log File Not Found"

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                msgs = data if isinstance(data, list) else data.get("messages", [])
                if not msgs: return "Empty Log"
                
                target_idx = -1
                if critical_step is not None and str(critical_step).isdigit():
                    estimated_idx = (int(critical_step) * 2) - 1
                    if 0 <= estimated_idx < len(msgs): target_idx = estimated_idx
                    else: target_idx = len(msgs) - 1
                else: target_idx = len(msgs) - 1

                start = max(0, target_idx - 3)
                end = min(len(msgs), target_idx + 4)
                return json.dumps(msgs[start:end], indent=2)
        except: return "Error Reading Log"

    def _get_filtered_candidates(self, legacy_module):
        legacy_key = str(legacy_module).lower().strip()
        allowed_artifacts = self.BRIDGE_MAP.get(legacy_key, [])
        if not allowed_artifacts:
            if "plan" in legacy_key: allowed_artifacts = self.BRIDGE_MAP["planning"]
            elif "mem" in legacy_key: allowed_artifacts = self.BRIDGE_MAP["memory"]
            elif "act" in legacy_key: allowed_artifacts = self.BRIDGE_MAP["action"]
            elif "sys" in legacy_key: allowed_artifacts = self.BRIDGE_MAP["system"]
            else: return self.taxonomy_db

        filtered = {}
        for key, details in self.taxonomy_db.items():
            if details.get("artifact") in allowed_artifacts:
                filtered[key] = details
        return filtered if filtered else self.taxonomy_db

    def _construct_prompt(self, old_item, log_snippet, filtered_candidates):
        old_module = old_item.get("critical_failure_module", "Unknown")
        old_step = old_item.get("critical_failure_step", "Unknown")
        legacy_def = self.LEGACY_DEFINITIONS.get(old_module.lower(), "General failure.")
        
        old_reason_str = "N/A"
        if "step_annotations" in old_item and old_item["step_annotations"]:
            def get_reason(d):
                if isinstance(d, dict):
                    if "reasoning" in d: return d["reasoning"]
                    for v in d.values():
                        res = get_reason(v)
                        if res: return res
                elif isinstance(d, list):
                    for i in d:
                        res = get_reason(i)
                        if res: return res
                return None
            found = get_reason(old_item["step_annotations"])
            if found: old_reason_str = found

        tax_desc_list = []
        for key, val in filtered_candidates.items():
            artifact = val.get("artifact", "Unknown")
            desc = val.get("definition", "")
            tax_desc_list.append(f'- [{artifact}] "{key}": {desc}')
        tax_str = "\n".join(tax_desc_list)

        prompt = f"""
        You are a Data Taxonomy Specialist.
        
        # LEGACY ERROR
        Type: **"{old_module.upper()}"** (Def: {legacy_def})
        Reason: "{old_reason_str}"
        Step: {old_step}

        # SHIELDA CANDIDATES (Choose ONE)
        You must choose a label from this list that corresponds to the Legacy Type.
        {tax_str}

        # LOG CONTEXT
        ```json
        {log_snippet}
        ```

        # INSTRUCTIONS
        Select the single best fit.
        Example: If Legacy is "SYSTEM" and reason is "step limit", choose "Step Limit Exhaustion".
        
        # OUTPUT (JSON)
        {{ "selected_exception": "Exact String", "analysis": "Reasoning..." }}
        """
        return prompt

    def process_dataset(self, dataset_name, old_gt_filename, limit=None):
        print(f"\n🚀 Processing {dataset_name} (Limit: {limit})...")
        old_path = os.path.join(self.old_gt_dir, old_gt_filename)
        if not os.path.exists(old_path): return

        with open(old_path, 'r') as f: old_data = json.load(f)

        new_data = []
        target_data = old_data[:limit] if limit else old_data
        
        for item in tqdm(target_data): 
            case_id = item.get("trajectory_id")
            if not case_id: continue
            
            step = item.get("critical_failure_step")
            old_module = item.get("critical_failure_module", "Unknown")
            
            # === 🔥 Hard Mapping Check ===
            old_reason_raw = str(item.get("step_annotations", "")).lower()
            hard_coded_l2 = None
            for k, v in self.HARD_MAPPINGS.items():
                if k in old_reason_raw:
                    hard_coded_l2 = v
                    break
            
            new_item = item.copy()
            final_l2 = "Unknown"
            analysis = "Auto-generated"

            if hard_coded_l2:
                final_l2 = hard_coded_l2
                analysis = "Mapped via hard-coded keyword rule."
            else:
                # Normal LLM Flow
                filtered_db = self._get_filtered_candidates(old_module)
                log_snippet = self._get_log_slice(dataset_name, case_id, step)
                prompt = self._construct_prompt(item, log_snippet, filtered_db)
                
                try:
                    res = self.llm.generate_json(
                        system_prompt="Strict Mapper. JSON Only.",
                        user_prompt=prompt, temperature=0.0
                    )
                    if isinstance(res, list): res = res[0] if res else {}
                    
                    raw_l2 = res.get("selected_exception", "Unknown")
                    analysis = res.get("analysis", "")
                    
                    if raw_l2 in filtered_db: final_l2 = raw_l2
                    elif raw_l2 in self.taxonomy_db: final_l2 = raw_l2 # Allow slight deviation if valid
                    else:
                        # 🔥 Lowered threshold to 0.3
                        matches = difflib.get_close_matches(raw_l2, list(filtered_db.keys()), n=1, cutoff=0.3)
                        if matches: final_l2 = matches[0]
                except Exception as e:
                    new_item["shielda_gt"] = {"error": str(e)}
                    new_data.append(new_item)
                    continue

            # Final Lookup
            if final_l2 in self.taxonomy_db:
                info = self.taxonomy_db[final_l2]
                final_l1 = info.get("artifact", "Unknown")
                phases = info.get("phases", ["Unknown"])
                phase_code = "RP" if "Reasoning" in phases else "E"
            else:
                final_l1, phase_code = "Unknown", "Unknown"

            new_item["shielda_gt"] = {
                "layer_1_artifact": final_l1,
                "layer_2_exception": final_l2,
                "error_phase": phase_code,
                "analysis": analysis
            }
            new_data.append(new_item)

        save_path = os.path.join(self.output_dir, f"{dataset_name}_labels.json")
        with open(save_path, 'w') as f: json.dump(new_data, f, indent=2)
        print(f"✅ Saved to {save_path}")

if __name__ == "__main__":
    refiner = GroundTruthRefiner()
    datasets = {
        "alfworld": "alfworld_labels.json",
        "webshop": "webshop_labels.json",
        "gaia": "gaia_labels.json"
    }
    TEST_LIMIT = 5 
    for ds, fname in datasets.items():
        refiner.process_dataset(ds, fname)