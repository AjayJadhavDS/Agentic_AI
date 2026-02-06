from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# web search Agent
web_search_agent = Agent(
        name = "Web Search Agent",
        role = "Search the web for the information",
        model = Groq(id = "llama-3.1-8b-instant"),
        tools = [DuckDuckGo()],
        instructions = "Always include the sources",
        show_tools_calls = True,
        markdown= True,
)

# financial agent
finance_agent = Agent(
        name = "Finance Agent",
        role = "",
        model = Groq(id = "llama-3.1-8b-instant"),
        tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals = True)],
        instructions = "Use tables to display the data",
        show_tools_calls = True,
        markdown= True,
)

# orchastror agent
multi_ai_agent = Agent(
    team = [web_search_agent, finance_agent],
    instructions = ["Always include the sources","Use tables to display the data"],
    show_tools_calls = True,
    markdown= True,
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA", stream = True)
