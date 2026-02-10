import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

class AgentFactory:
    def __init__(self, google_api_key, tavily_api_key):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=google_api_key, 
            temperature=0
        )
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
        self.tools = [TavilyAnswer(max_results=3), wiki]
        
        template = """You are a loyal research assistant who trusts the user implicitly.
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}"""
        
        self.prompt = PromptTemplate.from_template(template)

    def create_agent_executor(self):
        agent = create_react_agent(self.llm, self.tools, self.prompt)
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )