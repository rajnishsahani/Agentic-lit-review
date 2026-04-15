import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.shared_state import AgentState

batch_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant that synthesizes academic papers.
    Given a list of papers with titles and abstracts:
    1. Write a 2-3 sentence summary for EACH paper
    2. Then identify 3-5 major themes across all papers
    
    Format your response EXACTLY like this:
    SUMMARIES:
    Paper 1: [summary]
    Paper 2: [summary]
    ...
    THEMES:
    ["theme 1", "theme 2", "theme 3"]"""),
    ("human", """Papers to synthesize:

{papers_text}

Provide summaries and themes:""")
])

def synthesis_agent(state: AgentState) -> AgentState:
    """Summarizes papers and identifies themes — single batched Gemini call."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    print(f"\n[Synthesis Agent] Synthesizing {len(state['screened_papers'])} papers...")
    
    # Build one prompt with all papers
    papers_text = ""
    for i, paper in enumerate(state["screened_papers"], 1):
        papers_text += f"Paper {i}: {paper['title']}\nAbstract: {paper['abstract'][:400]}\n\n"
    
    time.sleep(3)
    chain = batch_prompt | llm
    response = chain.invoke({"papers_text": papers_text})
    content = response.content
    
    # Parse summaries and themes
    summaries = []
    if "SUMMARIES:" in content and "THEMES:" in content:
        summary_section = content.split("THEMES:")[0].replace("SUMMARIES:", "").strip()
        theme_section = content.split("THEMES:")[1].strip()
        
        # Assign summaries to papers
        summary_lines = [line.strip() for line in summary_section.split("\n") if line.strip()]
        for i, paper in enumerate(state["screened_papers"]):
            if i < len(summary_lines):
                summary = summary_lines[i]
                if summary.startswith(f"Paper {i+1}:"):
                    summary = summary[len(f"Paper {i+1}:"):].strip()
                paper["summary"] = summary
                summaries.append(f"- {paper['title']}: {summary}")
            else:
                paper["summary"] = "Summary not generated"
                summaries.append(f"- {paper['title']}: Summary not generated")
        
        # Parse themes
        try:
            themes = eval(theme_section.strip())
        except:
            themes = ["General AI Research"]
    else:
        # Fallback if format wasn't followed
        for paper in state["screened_papers"]:
            paper["summary"] = "Summary pending"
            summaries.append(f"- {paper['title']}: Summary pending")
        themes = ["General AI Research"]
    
    print(f"[Synthesis Agent] Identified themes: {themes}")
    
    return {
        **state,
        "summaries": summaries,
        "themes": themes,
        "screened_papers": state["screened_papers"],
        "messages": state["messages"] + [f"Synthesis complete: {len(themes)} themes identified"]
    }