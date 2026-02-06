from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import statistics
import json
import os
from dotenv import load_dotenv
load_dotenv()

# ======================================================
# TOOL SETUP
# ======================================================

#yfinance = YFinanceTools(historical_prices=True)
news_tool = DuckDuckGo()

fx_trend_agent = Agent(
    name="FX Trend Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=[
        "You are an FX trend analyst.",
        "You will be given a remittance corridor as input.",
        "Assess whether the short-term exchange rate trend for this corridor is",
        "Favorable, Unfavorable, or Uncertain for sending money.",
        "Do not assume any specific countries unless mentioned in the input.",
        "Do not use numbers.",
        "Start your response with exactly one word:",
        "Favorable, Unfavorable, or Uncertain.",
        "Then explain your reasoning in 1–2 simple sentences."
    ],
    markdown=True
)

# ======================================================
# AGENT 2: NEWS & MACRO SENTIMENT AGENT
# ======================================================
# This agent looks at recent news and macro signals
# that may impact the destination currency.
news_sentiment_agent = Agent(
    name="News Sentiment Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=[
        "You analyze recent economic and macro trends affecting a remittance corridor.",
        "Base your analysis on general recent global economic conditions.",
        "Focus on inflation, interest rates, political stability, and volatility.",
        "Do NOT call any external tools.",
        "Summarize sentiment as:",
        "Positive, Negative, or Uncertain.",
        "Use very simple language."
    ],
    markdown=True
)

# ======================================================
# AGENT 3: ORCHESTRATOR / SMART SEND MANAGER
# ======================================================
# This agent combines insights from other agents
# and gives the final recommendation to the user.

orchestrator_agent = Agent(
    name="Smart Send Orchestrator",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=[
        "You are a remittance advisor inside a money transfer app.",
        "You receive:",
        "1) FX trend analysis",
        "2) News sentiment analysis",
        "",
        "Decision rules:",
        "- If FX is Favorable AND sentiment is Positive → Send Now",
        "- If FX is Unfavorable OR sentiment is Negative → Consider Waiting",
        "- If anything is Uncertain → Be cautious and suggest waiting",
        "",
        "Speak directly to the customer.",
        "Be warm, reassuring, and easy to understand.",
        "End with a clear recommendation:",
        "Send Now or Consider Waiting."
    ],
    markdown=True
)

# ======================================================
# ORCHESTRATION FLOW (MAIN FUNCTION)
# ======================================================

def smart_send_recommendation(corridor: str):
    """
    Takes a remittance corridor (e.g. 'US to Mexico')
    and returns a final recommendation for the user.
    """

    # ---- Agent 1: FX Trend Reasoning ----
    fx_result = fx_trend_agent.run(
        f"Analyze FX trend for corridor: {corridor}"
    ).content

    # ---- Agent 2: News & Macro Sentiment ----
    news_result = news_sentiment_agent.run(
        f"Analyze recent news affecting this corridor: {corridor}"
    ).content

    # ---- Agent 3: Orchestrator / Final Decision ----
    final_recommendation = orchestrator_agent.run(
        f"""
        FX Trend Analysis:
        {fx_result}

        News Sentiment Analysis:
        {news_result}
        """
    ).content

    return final_recommendation

# ======================================================
# RUN DEMO
# ======================================================

if __name__ == "__main__":
    corridor_input = "US to INDIA"
    result = smart_send_recommendation(corridor_input)

    print("\n=== SMART SEND RECOMMENDATION ===\n")
    print(result)
