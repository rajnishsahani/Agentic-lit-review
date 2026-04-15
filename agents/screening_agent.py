import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.shared_state import AgentState, Paper
from typing import List

batch_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research paper screening assistant.
    Given a research topic and a list of papers with their titles and abstracts,
    rate EACH paper's relevance on a scale of 0.0 to 1.0.
    Return ONLY a Python list of floats in the same order as the papers.
    Example for 4 papers: [0.9, 0.3, 0.7, 0.6]"""),
    ("human", """Research topic: {topic}

{papers_text}

Relevance scores (Python list of floats, one per paper):""")
])

def screening_agent(state: AgentState) -> AgentState:
    """Screens papers for relevance — single batched Gemini call."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    print(f"\n[Screening Agent] Screening {len(state['raw_papers'])} papers...")
    
    # Build one big prompt with all papers
    papers_text = ""
    for i, paper in enumerate(state["raw_papers"], 1):
        papers_text += f"Paper {i}: {paper['title']}\nAbstract: {paper['abstract'][:300]}\n\n"
    
    time.sleep(3)
    chain = batch_prompt | llm
    response = chain.invoke({
        "topic": state["research_topic"],
        "papers_text": papers_text
    })
    
    # Parse scores
    try:
        scores = eval(response.content.strip())
    except:
        scores = [0.5] * len(state["raw_papers"])
    
    # Assign scores and filter
    screened_papers: List[Paper] = []
    for i, paper in enumerate(state["raw_papers"]):
        score = scores[i] if i < len(scores) else 0.5
        paper["relevance_score"] = score
        if score >= 0.5:
            screened_papers.append(paper)
    
    screened_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    print(f"[Screening Agent] {len(screened_papers)} papers passed screening")
    
    return {
        **state,
        "screened_papers": screened_papers,
        "messages": state["messages"] + [f"Screening complete: {len(screened_papers)} relevant papers"]
    }