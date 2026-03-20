import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state.shared_state import AgentState, Paper
from typing import List

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research paper screening assistant. 
    Given a research topic and a paper's title and abstract, 
    rate the paper's relevance on a scale of 0.0 to 1.0.
    Return ONLY a number between 0.0 and 1.0, nothing else."""),
    ("human", """Research topic: {topic}
    
Paper title: {title}
Paper abstract: {abstract}

Relevance score (0.0 to 1.0):""")
])

def screening_agent(state: AgentState) -> AgentState:
    """Screens papers for relevance to the research topic."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
    print(f"\n[Screening Agent] Screening {len(state['raw_papers'])} papers...")
    
    chain = prompt | llm
    screened_papers: List[Paper] = []
    
    for paper in state["raw_papers"]:
        try:
            response = chain.invoke({
                "topic": state["research_topic"],
                "title": paper["title"],
                "abstract": paper["abstract"][:500]
            })
            score = float(response.content.strip())
        except:
            score = 0.5
        
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