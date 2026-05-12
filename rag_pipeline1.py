import os
import time

from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    SystemMessage
)

from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=500
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# =========================================================
# TOOLS
# =========================================================

@tool
def retrieve_financial_data(query: str) -> str:
    """
    Retrieve relevant financial intelligence data.
    """

    docs = retriever.invoke(query)
    if not docs:
        return "No relevant financial data found."

    return "\n\n".join([
        doc.page_content for doc in docs
    ])


@tool
def retrieve_market_analysis(query: str) -> str:
    """
    Retrieve revenue, market growth,
    stock performance and financial trends.
    """

    docs = retriever.invoke(
        "market growth revenue performance " + query
    )
    return "\n\n".join([
        doc.page_content for doc in docs
    ])


@tool
def retrieve_portfolio_data(query: str) -> str:
    """
    Retrieve portfolio strategy,
    diversification and investment insights.
    """

    docs = retriever.invoke(
        "portfolio investment diversification " + query
    )
    return "\n\n".join([
        doc.page_content for doc in docs
    ])


@tool
def retrieve_risk_data(query: str) -> str:
    """
    Retrieve risk intelligence,
    debt exposure and volatility analysis.
    """

    docs = retriever.invoke(
        "risk volatility debt exposure " + query
    )
    return "\n\n".join([
        doc.page_content for doc in docs
    ])

# =========================================================
# TOOL LIST
# =========================================================

tools = [
    retrieve_financial_data,
    retrieve_market_analysis,
    retrieve_portfolio_data,
    retrieve_risk_data
]

tool_map = {
    tool.name: tool for tool in tools
}


# TOOL BINDING
llm_with_tools = llm.bind_tools(tools)

# STATE
class FinancialState(TypedDict):
    query: str
    mode: str
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]
    context: str
    retrieval_analysis: str
    market_analysis: str
    portfolio_analysis: str
    risk_analysis: str
    validation_status: str
    final_response: str
    

# TOOL EXECUTOR NODE
def execute_tools(state: FinancialState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_outputs = []
    collected_context = []
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            selected_tool = tool_map[tool_name]
            output = selected_tool.invoke(tool_args)
            collected_context.append(output)
            tool_outputs.append(
                ToolMessage(
                    content=output,
                    tool_call_id=tool_call["id"]
                )
            )

    combined_context = "\n\n".join(collected_context)
    return {
        "messages": tool_outputs,
        "context": combined_context
    }

# =========================================================
# RETRIEVAL AGENT
# =========================================================

retrieval_prompt = """
You are a Financial Retrieval Intelligence Agent.

Your ONLY responsibility:
- retrieve financial evidence
- retrieve earnings data
- retrieve stock intelligence
- retrieve revenue insights
- retrieve business signals

MANDATORY:
Use the most appropriate retrieval tool.

User Query:
{query}
"""


def retrieval_agent(state: FinancialState):
    prompt = retrieval_prompt.format(
        query=state["query"]
    )
    response = llm_with_tools.invoke([
        HumanMessage(content=prompt)
    ])
    return {
        "messages": [response]
    }

# =========================================================
# ROUTER
# =========================================================

def should_continue(state: FinancialState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "continue"

# =========================================================
# MARKET ANALYST
# =========================================================

market_prompt = """
You are a Senior Wall Street Financial Market Analyst.

You specialize in:
- equity research
- institutional market analysis
- revenue trend forecasting
- investment intelligence
- macroeconomic analysis

USER QUERY:
{query}

RETRIEVED FINANCIAL CONTEXT:
{context}

TASKS:
1. Analyze financial performance
2. Analyze growth indicators
3. Analyze revenue trends
4. Analyze investor confidence
5. Identify strengths and weaknesses
6. Analyze future opportunities
7. Analyze institutional signals

IMPORTANT:
- Use ONLY retrieved evidence
- Mention exact financial signals when available
- Avoid hallucinations
- Keep analysis professional

OUTPUT FORMAT:
## Executive Summary
## Revenue Analysis
## Growth Indicators
## Business Strengths
## Market Outlook
"""


def market_analyst_agent(state: FinancialState):
    prompt = market_prompt.format(
        query=state["query"],
        context=state["context"]
    )
    response = llm.invoke(prompt)
    return {
        "market_analysis": response.content
    }

# =========================================================
# PORTFOLIO AGENT
# =========================================================

portfolio_prompt = """
You are an Institutional Portfolio Strategy AI Agent.

Your responsibilities:
- portfolio allocation
- diversification strategy
- investment recommendation
- risk-adjusted allocation
- long-term investment analysis

USER QUERY:
{query}

MARKET ANALYSIS:
{market_analysis}

TASKS:
1. Determine BUY / HOLD / SELL
2. Evaluate risk vs reward
3. Analyze diversification opportunities
4. Evaluate institutional investment potential
5. Analyze long-term outlook

IMPORTANT:
- Use evidence-based reasoning only
- Avoid hallucinations
- Mention financial indicators when possible

OUTPUT FORMAT:
## Portfolio Recommendation
## Risk vs Reward
## Diversification Strategy
## Institutional Outlook
## Final Decision
"""


def portfolio_agent(state: FinancialState):
    prompt = portfolio_prompt.format(
        query=state["query"],
        market_analysis=state["market_analysis"]
    )
    response = llm.invoke(prompt)
    return {
        "portfolio_analysis": response.content
    }

# =========================================================
# RISK AGENT
# =========================================================

risk_prompt = """
You are an Institutional Financial Risk Intelligence Agent.

You specialize in:
- volatility analysis
- debt exposure
- liquidity risk
- operational risk
- macroeconomic threats
- institutional risk scoring

USER QUERY:
{query}

MARKET ANALYSIS:
{market_analysis}

PORTFOLIO ANALYSIS:
{portfolio_analysis}

TASKS:
1. Identify financial risks
2. Analyze market volatility
3. Analyze debt exposure
4. Evaluate operational uncertainty
5. Generate institutional risk score

RISK SCORE:
1 = Very Low Risk
10 = Extremely High Risk

OUTPUT FORMAT:
## Risk Summary
## Financial Threats
## Volatility Analysis
## Debt Exposure
"""


def risk_agent(state: FinancialState):
    prompt = risk_prompt.format(
        query=state["query"],
        market_analysis=state["market_analysis"],
        portfolio_analysis=state["portfolio_analysis"]
    )
    response = llm.invoke(prompt)
    return {
        "risk_analysis": response.content
    }
    
# =========================================================
# FINAL AGENT
# =========================================================

final_prompt = PromptTemplate(
    input_variables=[
        "query",
        "mode",
        "market",
        "portfolio",
        "risk"
    ],
    template="""
You are the Chief Financial Intelligence Officer AI.
Generate a professional institutional-level report.

USER QUERY:
{query}

MODE:
{mode}

MARKET ANALYSIS:
{market}

PORTFOLIO ANALYSIS:
{portfolio}

RISK ANALYSIS:
{risk}

If mode is simple:
- answer briefly
- avoid markdown headings

If mode is detailed:
- provide structured institutional analysis
- include risks and opportunities

OUTPUT FORMAT:

## Executive Summary
## Financial Insights
## Investment Recommendation
## Risk Assessment
## Strategic Outlook
## Confidence Score
"""
)

def final_agent(state: FinancialState):

    if state["mode"] == "simple":

        prompt = f"""
You are a financial assistant.

RULES:
- Give a short answer (max 8-10 lines)
- No headings
- No deep explanation
- Only key insight and recommendation

User Query:
{state["query"]}

Market:
{state["market_analysis"]}

Portfolio:
{state["portfolio_analysis"]}

Risk:
{state["risk_analysis"]}

Answer:
"""

    else:  # detailed mode

        prompt = f"""
You are a Chief Financial Intelligence Officer.

Write a FULL institutional report with structure:

## Executive Summary
## Financial Insights
## Investment Recommendation
## Risk Assessment
## Strategic Outlook
## Confidence Score

User Query:
{state["query"]}

Market:
{state["market_analysis"]}

Portfolio:
{state["portfolio_analysis"]}

Risk:
{state["risk_analysis"]}

Be detailed, analytical, and structured.
"""

    response = llm.invoke(prompt)

    return {
        "final_response": response.content
    }

# GRAPH
workflow = StateGraph(FinancialState)

# NODES
workflow.add_node("RetrieverAgent",retrieval_agent)
workflow.add_node("ToolExecutor",execute_tools)
workflow.add_node("MarketAnalystAgent",market_analyst_agent)
workflow.add_node("PortfolioAgent",portfolio_agent)
workflow.add_node("RiskAgent",risk_agent)
workflow.add_node("FinalAgent",final_agent)

# ENTRY
workflow.set_entry_point("RetrieverAgent")

# CONDITIONAL TOOL FLOW
workflow.add_conditional_edges(
    "RetrieverAgent",
    should_continue,
    {
        "tools": "ToolExecutor",
        "continue": "MarketAnalystAgent"
    }
)

workflow.add_edge("ToolExecutor","MarketAnalystAgent")

# MAIN FLOW
workflow.add_edge("MarketAnalystAgent","PortfolioAgent")
workflow.add_edge("PortfolioAgent","RiskAgent")
workflow.add_edge("RiskAgent","FinalAgent")
workflow.add_edge("FinalAgent",END)

# COMPILE
graph = workflow.compile()

# NORMAL EXECUTION
def run_financial_rag(
    query: str,
    mode: str = "detailed",
    evaluation: bool = False
):
    # EVALUATION MODE
    if evaluation:
        start = time.time()
        # Direct retrieval
        docs = retriever.invoke(query)
        context = "\n\n".join([
            doc.page_content
            for doc in docs
        ])

        # Simple QA Prompt
        prompt = f"""
        You are an extraction system.

        Extract ONLY the exact factual answer from the context.

        Rules:
        - Output ONLY one sentence
        - Do NOT explain
        - Do NOT summarize
        - Do NOT add commentary
        - Preserve exact numbers and names
        - If answer is not found, say: Not found

        Context:
        {context}

        Question:
        {query}

        Exact Answer:
        """

        response = llm.invoke(prompt)
        end = time.time()
        latency = round(
            end - start,
            2
        )
        return {
            "answer": response.content,
            "latency": latency
        }

    # NORMAL AGENTIC MODE
    start = time.time()
    response = graph.invoke({
        "query": query,
        "mode": mode,
        "messages": []
    })
    end = time.time()
    latency = round(
        end - start,
        2
    )
    return {
        "answer": response["final_response"],
        "latency": latency
    }

# STREAMING EXECUTION
def run_financial_rag_stream(
    query: str,
    mode: str = "detailed"
):
    initial_state = {
        "query": query,
        "mode": mode,
        "messages": []
    }
    final_state = None
    for event in graph.stream(initial_state):
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        final_state = node_output
        preview = " "
        
        # TOOL RETRIEVAL PREVIEW
        if (
            isinstance(node_output, dict)
            and "messages" in node_output
        ):
            preview = "Financial documents retrieved."

        # MARKET ANALYSIS
        if (
            isinstance(node_output, dict)
            and "market_analysis" in node_output
        ):
            preview = (
                node_output["market_analysis"][:250]
            )

        # PORTFOLIO ANALYSIS
        elif (
            isinstance(node_output, dict)
            and "portfolio_analysis" in node_output
        ):
            preview = (
                node_output["portfolio_analysis"][:250]
            )

        # RISK ANALYSIS
        elif (
            isinstance(node_output, dict)
            and "risk_analysis" in node_output
        ):
            preview = (
                node_output["risk_analysis"][:250]
            )

        # FINAL REPORT
        elif (
            isinstance(node_output, dict)
            and "final_response" in node_output
        ):

            preview = (
                "Final institutional report generated."
            )

        yield {
            "agent": node_name,
            "preview": preview
        }

    yield {
        "final_answer": final_state["final_response"]
    }


# TEST

if __name__ == "__main__":

    query = """Enter your Query"""

    result = run_financial_rag(query)

    print("\n")
    print("=" * 80)
    print("FINAL FINANCIAL REPORT")
    print("=" * 80)
    print("\n")
    print(result["answer"])
    print("\n")
    print("=" * 80)