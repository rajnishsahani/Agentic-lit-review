from typing import TypedDict, List, Optional

class Paper(TypedDict):
    title: str
    authors: List[str]
    abstract: str
    year: int
    url: str
    relevance_score: Optional[float]
    summary: Optional[str]
    themes: Optional[List[str]]

class AgentState(TypedDict):
    # Input
    research_topic: str
    
    # Search Agent outputs
    search_queries: List[str]
    raw_papers: List[Paper]
    
    # Screening Agent outputs
    screened_papers: List[Paper]
    
    # Synthesis Agent outputs
    summaries: List[str]
    themes: List[str]
    
    # Planning Agent outputs
    research_gaps: List[str]
    research_plan: str
    reading_order: List[dict]
    
    # Controller
    iteration: int
    max_iterations: int
    status: str
    messages: List[str]