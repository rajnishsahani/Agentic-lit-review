import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.shared_state import AgentState
from tools.arxiv_tool import search_arxiv
from tools.semantic_scholar import search_semantic_scholar

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant that generates effective search queries 
    for academic literature review. Given a research topic, generate 3 diverse and 
    specific search queries to find relevant papers on arXiv.
    Return ONLY a Python list of 3 strings, nothing else.
    Example: ["query 1", "query 2", "query 3"]"""),
    ("human", "Research topic: {topic}")
])

def search_agent(state: AgentState) -> AgentState:
    """Generates search queries and retrieves papers from arXiv and Semantic Scholar."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    print(f"\n[Search Agent] Searching for: {state['research_topic']}")
    
    time.sleep(3)
    chain = prompt | llm
    response = chain.invoke({"topic": state["research_topic"]})
    
    try:
        queries = eval(response.content)
    except:
        queries = [state["research_topic"]]
    
    print(f"[Search Agent] Generated queries: {queries}")
    
    all_papers = []
    seen_titles = set()
    
    # Search arXiv
    for query in queries:
        papers = search_arxiv(query, max_results=5)
        for paper in papers:
            if paper["title"] not in seen_titles:
                seen_titles.add(paper["title"])
                all_papers.append(paper)
    
    # Search Semantic Scholar
    for query in queries:
        papers = search_semantic_scholar(query, max_results=5)
        for paper in papers:
            if paper["title"] not in seen_titles:
                seen_titles.add(paper["title"])
                all_papers.append(paper)
    
    print(f"[Search Agent] Found {len(all_papers)} unique papers (arXiv + Semantic Scholar)")
    
    return {
        **state,
        "search_queries": queries,
        "raw_papers": all_papers,
        "messages": state["messages"] + [f"Search complete: {len(all_papers)} papers found"]
    }