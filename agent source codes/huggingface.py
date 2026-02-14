import os
import requests
from smolagents import CodeAgent, LiteLLMModel, tool

class HFAgentFactory:
    def __init__(self, google_api_key, tavily_api_key):

        os.environ["GEMINI_API_KEY"] = google_api_key
        self.model = LiteLLMModel(
            model_id="gemini/gemini-2.0-flash",
            temperature=0
        )
        self.tavily_api_key = tavily_api_key
        
        self.system_prompt = """You are a highly efficient academic researcher. 
GUIDELINES:
1. **Prioritize Prominence**: When a query involves a world-famous figure, prioritize their primary historical residence.
2. **Efficiency Bias**: If the top search results link a person to a location during a specific year, assume that is definitive.
3. **Consensus over Detail**: Follow the general historical consensus.
IMPORTANT: When calling tools, ensure argument types are correct (e.g., years must be Integers).
"""

    def create_agent_executor(self):
        
        @tool
        def arxiv_search(query: str, year: int) -> str:
            """
            Search for academic papers on ArXiv.
            Args:
                query: The search topic.
                year: The publication year. MUST be an integer (e.g., 2012).
            """

            try:
                _ = year - 1900
            except TypeError as e:

                raise e
                
            return f"Found papers for '{query}' published in {year}."

        @tool
        def tavily_search(query: str) -> str:
            """
            Search the web using Tavily API.
            Args:
                query: The search query.
            """
            try:
                resp = requests.post(
                    "https://api.tavily.com/search",
                    json={"api_key": self.tavily_api_key, "query": query, "max_results": 3}
                )
                return str(resp.json().get('results', "No results."))
            except Exception as e:
                return f"Search error: {str(e)}"

        return CodeAgent(
            model=self.model,
            tools=[arxiv_search, tavily_search],
            system_prompt=self.system_prompt,
            add_base_tools=True,
            max_steps=5
        )