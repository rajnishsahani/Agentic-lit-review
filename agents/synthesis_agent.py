import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state.shared_state import AgentState

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant that summarizes academic papers.
    Given a paper title and abstract, write a 2-3 sentence summary highlighting
    the key contribution and methodology."""),
    ("human", """Title: {title}
Abstract: {abstract}

Summary:""")
])

theme_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant that identifies themes across papers.
    Given a list of paper summaries, identify 3-5 major themes.
    Return ONLY a Python list of theme strings, nothing else.
    Example: ["theme 1", "theme 2", "theme 3"]"""),
    ("human", "Paper summaries:\n{summaries}")
])

def synthesis_agent(state: AgentState) -> AgentState:
    """Summarizes papers and identifies common themes."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
    print(f"\n[Synthesis Agent] Synthesizing {len(state['screened_papers'])} papers...")
    
    summary_chain = summary_prompt | llm
    summaries = []
    
    for paper in state["screened_papers"]:
        response = summary_chain.invoke({
            "title": paper["title"],
            "abstract": paper["abstract"][:500]
        })
        paper["summary"] = response.content
        summaries.append(f"- {paper['title']}: {response.content}")
    
    theme_chain = theme_prompt | llm
    theme_response = theme_chain.invoke({
        "summaries": "\n".join(summaries[:10])
    })
    
    try:
        themes = eval(theme_response.content)
    except:
        themes = ["General AI Research"]
    
    print(f"[Synthesis Agent] Identified themes: {themes}")
    
    return {
        **state,
        "summaries": summaries,
        "themes": themes,
        "screened_papers": state["screened_papers"],
        "messages": state["messages"] + [f"Synthesis complete: {len(themes)} themes identified"]
    }