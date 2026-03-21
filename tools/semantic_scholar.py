import requests
from typing import List
from state.shared_state import Paper

BASE_URL = "https://api.semanticscholar.org/graph/v1"

FIELDS = "title,authors,abstract,year,externalIds,openAccessPdf"

def search_semantic_scholar(query: str, max_results: int = 10) -> List[Paper]:
    """
    Search Semantic Scholar for papers matching the query.
    No API key required for basic usage.
    """
    url = f"{BASE_URL}/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": FIELDS
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Semantic Scholar] Request failed: {e}")
        return []

    papers = []
    for item in data.get("data", []):
        abstract = item.get("abstract") or ""
        if not abstract:
            continue

        authors = [a.get("name", "") for a in item.get("authors", [])]
        
        # Build URL from DOI or Semantic Scholar ID
        ext_ids = item.get("externalIds") or {}
        doi = ext_ids.get("DOI")
        paper_id = item.get("paperId", "")
        url_link = f"https://doi.org/{doi}" if doi else f"https://www.semanticscholar.org/paper/{paper_id}"

        paper: Paper = {
            "title": item.get("title", "Unknown Title"),
            "authors": authors,
            "abstract": abstract,
            "year": item.get("year") or 0,
            "url": url_link,
            "relevance_score": None,
            "summary": None,
            "themes": None
        }
        papers.append(paper)

    print(f"[Semantic Scholar] Found {len(papers)} papers for: {query}")
    return papers


def search_by_author(author_name: str, max_results: int = 5) -> List[Paper]:
    """
    Search papers by a specific author name.
    Useful for finding seminal works by key researchers.
    """
    url = f"{BASE_URL}/author/search"
    params = {"query": author_name, "fields": "name,papers.title,papers.abstract,papers.year"}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Semantic Scholar] Author search failed: {e}")
        return []

    papers = []
    for author in data.get("data", [])[:1]:
        for p in author.get("papers", [])[:max_results]:
            if not p.get("abstract"):
                continue
            paper: Paper = {
                "title": p.get("title", ""),
                "authors": [author_name],
                "abstract": p.get("abstract", ""),
                "year": p.get("year") or 0,
                "url": "",
                "relevance_score": None,
                "summary": None,
                "themes": None
            }
            papers.append(paper)

    return papers