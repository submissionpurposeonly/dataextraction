import json
import re

class EscalationController:
    def __init__(self, api_key):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    
    def perform_deep_analysis(self, history, taxonomy_context, ground_truth):
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_template()
        
        res = (prompt | self.llm).invoke({
            "history": str(history), 
            "taxonomy": taxonomy_context,
            "ground_truth": ground_truth
        })
        
        content = res.content
        try:
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            return json.loads(match.group(0)) if match else {"exception_name": "Unknown", "original_sin": "Analysis Failed", "strategic_prompt": "Stop."}
        except:
            return {"exception_name": "Unknown", "original_sin": "Analysis Failed", "strategic_prompt": "Stop."}
  