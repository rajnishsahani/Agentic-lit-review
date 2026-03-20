import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state.shared_state import AgentState

gaps_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research planning assistant.
    Given a research topic and themes found in existing literature,
    identify 3-5 research gaps or open questions.
    Return ONLY a Python list of strings, nothing else.
    Example: ["gap 1", "gap 2", "gap 3"]"""),
    ("human", """Research topic: {topic}
    
Themes in existing literature:
{themes}

Research gaps:""")
])

plan_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research planning assistant.
    Given a research topic, existing themes, and identified gaps,
    write a structured research plan with:
    1. Proposed research questions
    2. Suggested methodology
    3. Evaluation strategy
    Keep it concise and actionable."""),
    ("human", """Research topic: {topic}

Themes: {themes}

Research gaps: {gaps}

Research plan:""")
])

def planning_agent(state: AgentState) -> AgentState:
    """Identifies research gaps and generates a research plan."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
    print(f"\n[Planning Agent] Generating research plan...")
    
    gaps_chain = gaps_prompt | llm
    gaps_response = gaps_chain.invoke({
        "topic": state["research_topic"],
        "themes": "\n".join(state["themes"])
    })
    
    try:
        gaps = eval(gaps_response.content)
    except:
        gaps = ["Further research needed"]
    
    plan_chain = plan_prompt | llm
    plan_response = plan_chain.invoke({
        "topic": state["research_topic"],
        "themes": "\n".join(state["themes"]),
        "gaps": "\n".join(gaps)
    })
    
    print(f"[Planning Agent] Research plan generated")
    
    return {
        **state,
        "research_gaps": gaps,
        "research_plan": plan_response.content,
        "messages": state["messages"] + ["Planning complete: research plan generated"]
    }