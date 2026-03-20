import arxiv
from typing import List
from state.shared_state import Paper

def search_arxiv(query: str, max_results: int = 10) -> List[Paper]:
    """Search arXiv for papers matching the query."""
    client = arxiv.Client()
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in client.results(search):
        paper: Paper = {
            "title": result.title,
            "authors": [str(a) for a in result.authors],
            "abstract": result.summary,
            "year": result.published.year,
            "url": result.entry_id,
            "relevance_score": None,
            "summary": None,
            "themes": None
        }
        papers.append(paper)
    
    return papers