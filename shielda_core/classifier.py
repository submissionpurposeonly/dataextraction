import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class ExceptionClassifier:
    def __init__(self, api_key, taxonomy_context):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
        
        self.taxonomy_str = taxonomy_context 

    def diagnose(self, thought, action, observation, ground_truth, history_str=""):
        prompt = ChatPromptTemplate.from_template()

        res = (prompt | self.llm).invoke({
            "taxonomy": self.taxonomy_str,
            "ground_truth": ground_truth,
            "observation": str(observation)[:1000],
            "action": str(action),
            "thought": str(thought)
        })

        try:
            return json.loads(re.search(r'\{.*\}', res.content, re.DOTALL).group(0))
        except:
            return {"id": "None"}