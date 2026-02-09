import os
from smolagents import CodeAgent, LiteLLMModel, Tool
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# ✅ 将 LangChain 的工具包装成 smolagents 能识别的格式
class VeriGuardToolWrapper(Tool):
    def __init__(self, langchain_tool, name, description):
        super().__init__()
        self.tool = langchain_tool
        self.name = name
        self.description = description
        self.inputs = {"query": {"type": "string", "description": "The search query"}}
        self.output_type = "string"

    def forward(self, query: str) -> str:
        # 🛡️ 这里就是插入 VeriGuard 策略验证的地方！
        # print(f"VeriGuard Intercepting: calling {self.name} with {query}")
        return self.tool.run(query)

class HFAgentFactory:
    def __init__(self, google_api_key, tavily_api_key):
        # 1. 初始化模型 (使用 LiteLLM 桥接 Gemini)
        # 注意：需要安装 pip install smolagents litellm
        os.environ["GEMINI_API_KEY"] = google_api_key
        self.model = LiteLLMModel(
            model_id="gemini/gemini-2.0-flash",
            temperature=0
        )
        
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        
        # 2. 准备工具 (保持与你 LangChain 版一致)
        wiki_lc = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
        tavily_lc = TavilyAnswer(max_results=3)
        
        self.tools = [
            VeriGuardToolWrapper(tavily_lc, "tavily_search", "Search the web for real-time info."),
            VeriGuardToolWrapper(wiki_lc, "wikipedia", "Search Wikipedia for historical facts.")
        ]
        
        # 3. 设定系统 Prompt (对标你 LangChain 的 Template)
        self.system_prompt = """You are a highly efficient academic researcher. 
GUIDELINES:
1. **Prioritize Prominence**: Focus on primary historical residences for famous figures.
2. **Efficiency Bias**: Use current search results as definitive for specific years.
3. **Consensus over Detail**: Follow general historical consensus.
"""

    def create_agent_executor(self):
        # smolagents 的 CodeAgent 会自动处理循环逻辑
        return CodeAgent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            add_base_tools=False, # 只用我们定义的工具
            max_steps=5
        )

# 使用示例
# factory = HFAgentFactory(google_key, tavily_key)
# agent = factory.create_agent_executor()
# agent.run("Where was Einstein living in 1933?")