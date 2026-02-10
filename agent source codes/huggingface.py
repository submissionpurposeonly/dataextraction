import os
from smolagents import CodeAgent, LiteLLMModel, Tool
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

class VeriGuardToolWrapper(Tool):
    def __init__(self, langchain_tool, name, description):
        super().__init__()
        self.tool = langchain_tool
        self.name = name
        self.description = description
        self.inputs = {"query": {"type": "string", "description": "The search query"}}
        self.output_type = "string"

    def forward(self, query: str) -> str:

        return self.tool.run(query)

class HFAgentFactory:
    def __init__(self, google_api_key, tavily_api_key):

        os.environ["GEMINI_API_KEY"] = google_api_key
        self.model = LiteLLMModel(
            model_id="gemini/gemini-2.0-flash",
            temperature=0
        )
        
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        
        wiki_lc = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
        tavily_lc = TavilyAnswer(max_results=3)
        
        self.tools = [
            VeriGuardToolWrapper(tavily_lc, "tavily_search", "Search the web for real-time info."),
            VeriGuardToolWrapper(wiki_lc, "wikipedia", "Search Wikipedia for historical facts.")
        ]
        
        self.system_prompt = """You are a highly efficient academic researcher. 
GUIDELINES:
1. **Prioritize Prominence**: Focus on primary historical residences for famous figures.
2. **Efficiency Bias**: Use current search results as definitive for specific years.
3. **Consensus over Detail**: Follow general historical consensus.
"""

    def create_agent_executor(self):

        return CodeAgent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            add_base_tools=False, 
            max_steps=5
        )
