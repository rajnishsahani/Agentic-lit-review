from typing import List, Dict
from state.shared_state import Paper

def precision(screened_papers: List[Paper], relevant_threshold: float = 0.5) -> float:
    """
    Proportion of screened papers that are relevant.
    Relevance determined by relevance_score >= threshold.
    """
    if not screened_papers:
        return 0.0
    relevant = sum(1 for p in screened_papers if p.get("relevance_score", 0) >= relevant_threshold)
    return round(relevant / len(screened_papers), 4)

def coverage(screened_papers: List[Paper], known_papers: List[str]) -> float:
    """
    Proportion of known key papers that were retrieved.
    known_papers: list of known paper titles to check against.
    """
    if not known_papers:
        return 0.0
    retrieved_titles = [p["title"].lower() for p in screened_papers]
    found = sum(1 for title in known_papers if title.lower() in retrieved_titles)
    return round(found / len(known_papers), 4)

def api_efficiency(messages: List[str]) -> Dict:
    """
    Counts agent calls made during the run as a proxy for API efficiency.
    """
    search_calls = sum(1 for m in messages if "Search complete" in m)
    screening_calls = sum(1 for m in messages if "Screening complete" in m)
    synthesis_calls = sum(1 for m in messages if "Synthesis complete" in m)
    planning_calls = sum(1 for m in messages if "Planning complete" in m)
    total = search_calls + screening_calls + synthesis_calls + planning_calls
    
    return {
        "search_calls": search_calls,
        "screening_calls": screening_calls,
        "synthesis_calls": synthesis_calls,
        "planning_calls": planning_calls,
        "total_agent_calls": total
    }

def iteration_rate(iteration: int, max_iterations: int) -> float:
    """
    How many iterations were needed relative to maximum allowed.
    Lower is better — 1.0 means only one iteration needed.
    """
    return round(iteration / max_iterations, 4)

def screening_rate(raw_papers: List[Paper], screened_papers: List[Paper]) -> float:
    """
    Proportion of raw papers that passed screening.
    """
    if not raw_papers:
        return 0.0
    return round(len(screened_papers) / len(raw_papers), 4)

def generate_report(state: dict, known_papers: List[str] = []) -> Dict:
    """
    Generates a full evaluation report from the final agent state.
    """
    report = {
        "research_topic": state["research_topic"],
        "papers_retrieved": len(state["raw_papers"]),
        "papers_after_screening": len(state["screened_papers"]),
        "screening_rate": screening_rate(state["raw_papers"], state["screened_papers"]),
        "precision": precision(state["screened_papers"]),
        "coverage": coverage(state["screened_papers"], known_papers),
        "themes_identified": len(state["themes"]),
        "research_gaps_identified": len(state["research_gaps"]),
        "iterations_used": state["iteration"],
        "iteration_rate": iteration_rate(state["iteration"], state["max_iterations"]),
        "api_efficiency": api_efficiency(state["messages"]),
    }
    return report

def print_report(report: Dict):
    """Prints the evaluation report in a readable format."""
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"Topic          : {report['research_topic']}")
    print(f"Papers found   : {report['papers_retrieved']}")
    print(f"After screening: {report['papers_after_screening']}")
    print(f"Screening rate : {report['screening_rate']:.0%}")
    print(f"Precision      : {report['precision']:.0%}")
    print(f"Coverage       : {report['coverage']:.0%}")
    print(f"Themes found   : {report['themes_identified']}")
    print(f"Gaps identified: {report['research_gaps_identified']}")
    print(f"Iterations used: {report['iterations_used']}")
    print(f"\nAPI Efficiency:")
    for k, v in report['api_efficiency'].items():
        print(f"  {k}: {v}")