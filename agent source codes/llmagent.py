import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src.utils.file_manager import FileManager
from src.utils.llm_client import GeminiClient

class AgentSimulator:
    def __init__(self, benchmark_name="alfworld"):
        self.fm = FileManager()
        self.llm = GeminiClient()
        self.benchmark_name = benchmark_name
        try:
            self.system_prompt = self.fm.load_benchmark_prompt(f"agent_{benchmark_name}")
        except:
            self.system_prompt = "You are an autonomous AI agent. React to the environment."

    def step(self, history: list, injected_guidance: str = None) -> str:

        context_str = ""
        relevant_history = history[-15:] if len(history) > 15 else history
        for msg in relevant_history:
            role = msg.get('role', 'user').upper()
            content = msg.get('content', '')
            context_str += f"\n[{role}]: {content}"
        
        intervention_block = ""
        if injected_guidance:
            intervention_block = f"""
            \n#####################################################
            🛑 SYSTEM INSTRUCTION (Must Follow):
            {injected_guidance}
            #####################################################
            """

        full_prompt = f"""
        {self.system_prompt}
        
        === EXECUTION HISTORY ===
        {context_str}
        
        {intervention_block}
        
        YOUR TASK:
        Based on the history and the SYSTEM INSTRUCTION above, generate your **Next Action**.
        - Strictly follow the System Instruction.
        - Output the command directly.
        """

        try:
            return self._safe_call(full_prompt)
        except Exception as e:
            return f"Simulation Crashed: {str(e)}"

    def _safe_call(self, prompt_text):
        errors = []

        try:
            res = self.llm.generate_text(prompt_text)
            if hasattr(res, 'text'): return res.text.strip()
            return str(res).strip()
        except Exception as e:
            errors.append(f"Positional failed: {e}")

        try:
            res = self.llm.generate_text(prompt=prompt_text)
            if hasattr(res, 'text'): return res.text.strip()
            return str(res).strip()
        except Exception as e:
            errors.append(f"Keyword 'prompt' failed: {e}")

        try:
            if hasattr(self.llm, 'model'):
                res = self.llm.model.generate_content(prompt_text)
                return res.text.strip()
        except Exception as e:
            errors.append(f"Native generate_content failed: {e}")

        raise RuntimeError(f"All LLM call methods failed. Details: {errors}")