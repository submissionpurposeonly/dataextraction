import json
import re

class LocalHandler:
    def __init__(self, api_key, km):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
        self.km = km 

    def get_recovery_instruction(self, exception_id, diagnosis_thinking, raw_obs, retry_count):
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt_tmpl = ChatPromptTemplate.from_template()
    
        res = (prompt_tmpl | self.llm).invoke({
            "diagnosis_thinking": diagnosis_thinking,
            "observation": str(raw_obs)[:1000]
        })
        
        content = res.content
        if isinstance(content, list):
            content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])

        try:
            return json.loads(re.search(r'\{.*\}', content, re.DOTALL).group(0))
        except:
            return {"prompt": "Please re-verify the specific time and location constraints of the task."}