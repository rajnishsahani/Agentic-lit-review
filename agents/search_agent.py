import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state.shared_state import AgentState
from tools.arxiv_tool import search_arxiv
from tools.semantic_scholar import search_semantic_scholar

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant that generates effective search queries 
    for academic literature review. Given a research topic, generate 3 diverse and 
    specific search queries to find relevant papers.
    Return ONLY a Python list of 3 strings, nothing else.
    Example: ["query 1", "query 2", "query 3"]"""),
    ("human", "Research topic: {topic}")
])

def search_agent(state: AgentState) -> AgentState:
    """Generates search queries and retrieves papers from arXiv and Semantic Scholar."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
    print(f"\n[Search Agent] Searching for: {state['research_topic']}")
    
    chain = prompt | llm
    response = chain.invoke({"topic": state["research_topic"]})
    
    try:
        queries = eval(response.content)
    except:
        queries = [state["research_topic"]]
    
    print(f"[Search Agent] Generated queries: {queries}")
    
    all_papers = []
    seen_titles = set()
    
    for query in queries:
        # Search arXiv
        arxiv_papers = search_arxiv(query, max_results=5)
        for paper in arxiv_papers:
            if paper["title"] not in seen_titles:
                seen_titles.add(paper["title"])
                all_papers.append(paper)
        
        # Search Semantic Scholar
        ss_papers = search_semantic_scholar(query, max_results=5)
        for paper in ss_papers:
            if paper["title"] not in seen_titles:
                seen_titles.add(paper["title"])
                all_papers.append(paper)
    
    print(f"[Search Agent] Found {len(all_papers)} unique papers total")
    
    return {
        **state,
        "search_queries": queries,
        "raw_papers": all_papers,
        "messages": state["messages"] + [f"Search complete: {len(all_papers)} papers found"]
    }