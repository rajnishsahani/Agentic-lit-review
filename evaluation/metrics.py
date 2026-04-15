"""
Evaluation metrics for the Agentic Literature Review system.
Based on CIS 600 Week 9: Evaluating Agentic AI framework.

Metrics categories:
1. Tool Metrics (Selection Accuracy, Execution Success, Pass Rate)
2. Planning Metrics (Step Success Rate, Task Completion)
3. Retrieval & Screening Metrics (Coverage, Precision, Relevance)
4. Baseline Comparison (Agentic vs Non-Agentic)
5. Efficiency Metrics (API calls, latency, iterations)
"""

import time
from typing import Dict, List, Any
from tools.arxiv_tool import search_arxiv
from tools.semantic_scholar import search_semantic_scholar


# ============================================================
# 1. TOOL METRICS (Week 9 - Metrics for Tools)
# ============================================================

def tool_selection_accuracy(state: Dict) -> float:
    """
    Tool Selection Accuracy = # correct tools selected / Total tools required
    
    In our pipeline, the expected tools per step are:
    - Search Agent: arXiv API + Semantic Scholar API
    - Screening Agent: LLM LLM
    - Synthesis Agent: LLM LLM
    - Planning Agent: LLM LLM
    """
    required_tools = ["arxiv_api", "semantic_scholar_api", "llm"]
    used_tools = []
    
    if len(state.get("raw_papers", [])) > 0:
        # Check if papers came from arXiv (have arxiv URLs)
        arxiv_papers = [p for p in state["raw_papers"] if "arxiv" in p.get("url", "").lower()]
        if arxiv_papers:
            used_tools.append("arxiv_api")
        
        # Check if papers came from Semantic Scholar
        ss_papers = [p for p in state["raw_papers"] if "semanticscholar" in p.get("url", "").lower() 
                     or "arxiv" not in p.get("url", "").lower()]
        if ss_papers:
            used_tools.append("semantic_scholar_api")
    
    if len(state.get("screened_papers", [])) > 0:
        used_tools.append("llm")
    
    if not required_tools:
        return 0.0
    
    correct = len(set(used_tools) & set(required_tools))
    return correct / len(required_tools)


def tool_execution_success(state: Dict) -> Dict[str, Any]:
    """
    Tool Execution Success = Successful executions / Total tool calls
    
    Tracks whether each agent's tool calls succeeded or failed.
    """
    steps = {
        "search": {
            "executed": len(state.get("raw_papers", [])) > 0,
            "description": "arXiv + Semantic Scholar retrieval"
        },
        "screening": {
            "executed": len(state.get("screened_papers", [])) > 0,
            "description": "LLM relevance scoring"
        },
        "synthesis": {
            "executed": len(state.get("summaries", [])) > 0 and len(state.get("themes", [])) > 0,
            "description": "LLM summarization + theme extraction"
        },
        "planning": {
            "executed": len(state.get("research_gaps", [])) > 0 and len(state.get("research_plan", "")) > 0,
            "description": "LLM gap analysis + plan generation"
        }
    }
    
    total = len(steps)
    successful = sum(1 for s in steps.values() if s["executed"])
    
    return {
        "success_rate": successful / total if total > 0 else 0.0,
        "successful": successful,
        "total": total,
        "details": steps
    }


def pass_rate(results: List[Dict]) -> float:
    """
    Pass Rate = Successfully completed tasks / Total task attempts
    
    Evaluated across multiple runs with different topics.
    """
    if not results:
        return 0.0
    
    passed = sum(1 for r in results if 
                 len(r.get("screened_papers", [])) > 0 and
                 len(r.get("themes", [])) > 0 and
                 len(r.get("research_plan", "")) > 0)
    
    return passed / len(results)


# ============================================================
# 2. PLANNING METRICS (Week 9 - Metrics for Planning)
# ============================================================

def step_success_rate(state: Dict) -> Dict[str, Any]:
    """
    Step Success Rate = Successful steps / Total steps executed
    
    Pipeline steps: search → screen → synthesize → plan
    Each step is evaluated for successful completion.
    """
    steps = [
        ("Query Generation", len(state.get("search_queries", [])) > 0),
        ("Paper Retrieval", len(state.get("raw_papers", [])) > 0),
        ("Relevance Screening", len(state.get("screened_papers", [])) > 0),
        ("Summarization", len(state.get("summaries", [])) > 0),
        ("Theme Identification", len(state.get("themes", [])) > 0 and 
         state.get("themes", []) != ["General AI Research"]),
        ("Gap Analysis", len(state.get("research_gaps", [])) > 0 and 
         state.get("research_gaps", []) != ["Further research needed"]),
        ("Research Plan", len(state.get("research_plan", "")) > 50),
    ]
    
    total = len(steps)
    successful = sum(1 for _, passed in steps if passed)
    
    return {
        "rate": successful / total if total > 0 else 0.0,
        "successful": successful,
        "total": total,
        "steps": {name: "PASS" if passed else "FAIL" for name, passed in steps}
    }


def task_completion_rate(state: Dict) -> Dict[str, Any]:
    """
    Task Success Rate (Reasoning-Level) = Fully correct tasks / Total tasks
    
    A task is fully correct if ALL expected outputs are present and valid.
    """
    tasks = {
        "literature_search": (
            len(state.get("raw_papers", [])) >= 5,
            "Found at least 5 papers"
        ),
        "relevance_filtering": (
            len(state.get("screened_papers", [])) >= 3 and
            all(p.get("relevance_score", 0) >= 0.5 for p in state.get("screened_papers", [])),
            "At least 3 papers with relevance >= 0.5"
        ),
        "synthesis": (
            len(state.get("themes", [])) >= 3 and
            all(p.get("summary", "") != "" for p in state.get("screened_papers", [])),
            "All papers summarized + 3+ themes identified"
        ),
        "planning": (
            len(state.get("research_gaps", [])) >= 2 and
            len(state.get("research_plan", "")) > 100,
            "2+ gaps identified + detailed research plan"
        ),
    }
    
    total = len(tasks)
    completed = sum(1 for passed, _ in tasks.values() if passed)
    
    return {
        "rate": completed / total if total > 0 else 0.0,
        "completed": completed,
        "total": total,
        "tasks": {name: {"passed": passed, "criteria": desc} for name, (passed, desc) in tasks.items()}
    }


# ============================================================
# 3. RETRIEVAL & SCREENING METRICS (Week 5 - RAG Evaluation)
# ============================================================

def screening_precision(state: Dict) -> float:
    """
    Screening Precision = Papers with relevance >= 0.5 / Total screened papers
    
    Measures how accurately the screening agent filters relevant papers.
    """
    screened = state.get("screened_papers", [])
    if not screened:
        return 0.0
    
    relevant = sum(1 for p in screened if p.get("relevance_score", 0) >= 0.5)
    return relevant / len(screened)


def search_coverage(state: Dict) -> Dict[str, Any]:
    """
    Coverage = Papers retained after screening / Total papers found
    
    Measures the balance between finding enough papers and filtering quality.
    """
    raw = len(state.get("raw_papers", []))
    screened = len(state.get("screened_papers", []))
    
    return {
        "total_found": raw,
        "after_screening": screened,
        "retention_rate": screened / raw if raw > 0 else 0.0,
        "rejection_rate": (raw - screened) / raw if raw > 0 else 0.0,
    }


def source_diversity(state: Dict) -> Dict[str, Any]:
    """
    Measures diversity of paper sources (arXiv vs Semantic Scholar).
    """
    papers = state.get("raw_papers", [])
    if not papers:
        return {"arxiv": 0, "semantic_scholar": 0, "diversity_score": 0.0}
    
    arxiv = sum(1 for p in papers if "arxiv" in p.get("url", "").lower())
    semantic = len(papers) - arxiv
    
    total = len(papers)
    # Diversity is higher when sources are balanced
    if total == 0:
        diversity = 0.0
    else:
        ratio = min(arxiv, semantic) / max(arxiv, semantic) if max(arxiv, semantic) > 0 else 0.0
        diversity = ratio
    
    return {
        "arxiv": arxiv,
        "semantic_scholar": semantic,
        "total": total,
        "diversity_score": round(diversity, 3)
    }


def relevance_score_distribution(state: Dict) -> Dict[str, Any]:
    """
    Analyzes the distribution of relevance scores from screening.
    """
    screened = state.get("screened_papers", [])
    if not screened:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "high_relevance_count": 0}
    
    scores = [p.get("relevance_score", 0) for p in screened]
    
    return {
        "mean": round(sum(scores) / len(scores), 3),
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "high_relevance_count": sum(1 for s in scores if s >= 0.8),
        "medium_relevance_count": sum(1 for s in scores if 0.5 <= s < 0.8),
    }


# ============================================================
# 4. EFFICIENCY METRICS
# ============================================================

def efficiency_metrics(state: Dict, execution_time: float) -> Dict[str, Any]:
    """
    Measures efficiency of the pipeline.
    - API calls used (LLM calls)
    - Iterations used
    - Time per paper
    """
    total_papers = len(state.get("raw_papers", []))
    screened_papers = len(state.get("screened_papers", []))
    iterations = state.get("iteration", 1)
    
    # With batching: 1 call per agent = 4 LLM calls total
    llm_calls = 4 * iterations
    
    return {
        "total_execution_time_sec": round(execution_time, 2),
        "estimated_llm_calls": llm_calls,
        "iterations_used": iterations,
        "papers_per_second": round(total_papers / execution_time, 3) if execution_time > 0 else 0,
        "time_per_paper_sec": round(execution_time / total_papers, 2) if total_papers > 0 else 0,
        "screening_efficiency": round(screened_papers / total_papers, 3) if total_papers > 0 else 0,
    }


# ============================================================
# 5. BASELINE COMPARISON (Agentic vs Non-Agentic)
# ============================================================

def run_baseline(topic: str) -> Dict[str, Any]:
    """
    Non-agentic baseline: simple keyword search on arXiv only.
    No LLM screening, no synthesis, no planning.
    Just raw search results — what a researcher would get without the system.
    """
    start = time.time()
    
    # Simple keyword search — no query generation by LLM
    papers = search_arxiv(topic, max_results=10)
    
    elapsed = time.time() - start
    
    return {
        "topic": topic,
        "papers_found": len(papers),
        "papers": papers,
        "has_screening": False,
        "has_summaries": False,
        "has_themes": False,
        "has_gaps": False,
        "has_plan": False,
        "execution_time_sec": round(elapsed, 2),
        "llm_calls": 0,
    }


def compare_with_baseline(agentic_state: Dict, baseline: Dict, agentic_time: float) -> Dict[str, Any]:
    """
    Compares agentic pipeline output vs non-agentic baseline.
    This is the key comparison Professor Kumarawadu asked for.
    """
    agentic_papers = len(agentic_state.get("screened_papers", []))
    baseline_papers = baseline["papers_found"]
    
    comparison = {
        "metric": [],
        "agentic": [],
        "baseline": [],
        "improvement": []
    }
    
    # Papers found
    comparison["metric"].append("Total Papers Retrieved")
    comparison["agentic"].append(len(agentic_state.get("raw_papers", [])))
    comparison["baseline"].append(baseline_papers)
    comparison["improvement"].append("More" if len(agentic_state.get("raw_papers", [])) > baseline_papers else "Less")
    
    # Screened/Filtered papers
    comparison["metric"].append("Relevant Papers (Screened)")
    comparison["agentic"].append(agentic_papers)
    comparison["baseline"].append(f"{baseline_papers} (unfiltered)")
    comparison["improvement"].append("Quality filtered" if agentic_papers > 0 else "N/A")
    
    # Search diversity
    comparison["metric"].append("Search Sources")
    comparison["agentic"].append("arXiv + Semantic Scholar")
    comparison["baseline"].append("arXiv only")
    comparison["improvement"].append("Multi-source")
    
    # Query intelligence
    comparison["metric"].append("Query Strategy")
    comparison["agentic"].append(f"{len(agentic_state.get('search_queries', []))} LLM-generated queries")
    comparison["baseline"].append("1 keyword query")
    comparison["improvement"].append("Diverse coverage")
    
    # Summaries
    comparison["metric"].append("Paper Summaries")
    comparison["agentic"].append(f"{len(agentic_state.get('summaries', []))} generated")
    comparison["baseline"].append("None")
    comparison["improvement"].append("Automated synthesis")
    
    # Themes
    comparison["metric"].append("Theme Identification")
    comparison["agentic"].append(f"{len(agentic_state.get('themes', []))} themes")
    comparison["baseline"].append("None")
    comparison["improvement"].append("Cross-paper analysis")
    
    # Research gaps
    comparison["metric"].append("Research Gaps")
    comparison["agentic"].append(f"{len(agentic_state.get('research_gaps', []))} identified")
    comparison["baseline"].append("None")
    comparison["improvement"].append("Gap analysis")
    
    # Research plan
    comparison["metric"].append("Research Plan")
    comparison["agentic"].append("Generated" if agentic_state.get("research_plan", "") else "None")
    comparison["baseline"].append("None")
    comparison["improvement"].append("Structured planning")
    
    # Execution time
    comparison["metric"].append("Execution Time")
    comparison["agentic"].append(f"{round(agentic_time, 1)}s")
    comparison["baseline"].append(f"{baseline['execution_time_sec']}s")
    comparison["improvement"].append("Slower but comprehensive")
    
    # LLM calls
    comparison["metric"].append("LLM API Calls")
    comparison["agentic"].append("~4 (batched)")
    comparison["baseline"].append("0")
    comparison["improvement"].append("Intelligence cost")
    
    return comparison


# ============================================================
# 6. MASTER EVALUATION FUNCTION
# ============================================================

def evaluate_run(state: Dict, execution_time: float, topic: str) -> Dict[str, Any]:
    """
    Run all evaluation metrics on a pipeline result.
    
    Args:
        state: The final AgentState from the pipeline
        execution_time: Total pipeline execution time in seconds
        topic: The research topic used
    
    Returns:
        Dictionary with all evaluation results
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # --- Tool Metrics ---
    tool_acc = tool_selection_accuracy(state)
    tool_exec = tool_execution_success(state)
    
    print(f"\n--- TOOL METRICS ---")
    print(f"  Tool Selection Accuracy: {tool_acc:.2f}")
    print(f"  Tool Execution Success:  {tool_exec['success_rate']:.2f} ({tool_exec['successful']}/{tool_exec['total']})")
    for step, info in tool_exec["details"].items():
        status = "PASS" if info["executed"] else "FAIL"
        print(f"    {step}: {status} ({info['description']})")
    
    # --- Planning Metrics ---
    step_rate = step_success_rate(state)
    task_rate = task_completion_rate(state)
    
    print(f"\n--- PLANNING METRICS ---")
    print(f"  Step Success Rate:    {step_rate['rate']:.2f} ({step_rate['successful']}/{step_rate['total']})")
    for step, status in step_rate["steps"].items():
        print(f"    {step}: {status}")
    print(f"  Task Completion Rate: {task_rate['rate']:.2f} ({task_rate['completed']}/{task_rate['total']})")
    
    # --- Retrieval Metrics ---
    precision = screening_precision(state)
    coverage = search_coverage(state)
    diversity = source_diversity(state)
    relevance_dist = relevance_score_distribution(state)
    
    print(f"\n--- RETRIEVAL & SCREENING METRICS ---")
    print(f"  Screening Precision:  {precision:.2f}")
    print(f"  Coverage: {coverage['after_screening']}/{coverage['total_found']} papers retained ({coverage['retention_rate']:.2f})")
    print(f"  Source Diversity:     arXiv={diversity['arxiv']}, Semantic Scholar={diversity['semantic_scholar']} (score: {diversity['diversity_score']})")
    print(f"  Relevance Scores:    mean={relevance_dist['mean']}, min={relevance_dist['min']}, max={relevance_dist['max']}")
    
    # --- Efficiency Metrics ---
    efficiency = efficiency_metrics(state, execution_time)
    
    print(f"\n--- EFFICIENCY METRICS ---")
    print(f"  Execution Time:      {efficiency['total_execution_time_sec']}s")
    print(f"  LLM API Calls:    ~{efficiency['estimated_llm_calls']}")
    print(f"  Iterations Used:     {efficiency['iterations_used']}")
    print(f"  Time Per Paper:      {efficiency['time_per_paper_sec']}s")
    
    # --- Baseline Comparison ---
    print(f"\n--- BASELINE COMPARISON (Agentic vs Non-Agentic) ---")
    baseline = run_baseline(topic)
    comparison = compare_with_baseline(state, baseline, execution_time)
    
    for i in range(len(comparison["metric"])):
        print(f"  {comparison['metric'][i]}:")
        print(f"    Agentic:  {comparison['agentic'][i]}")
        print(f"    Baseline: {comparison['baseline'][i]}")
        print(f"    → {comparison['improvement'][i]}")
    
    print("\n" + "=" * 60)
    
    return {
        "tool_metrics": {
            "selection_accuracy": tool_acc,
            "execution_success": tool_exec,
        },
        "planning_metrics": {
            "step_success": step_rate,
            "task_completion": task_rate,
        },
        "retrieval_metrics": {
            "screening_precision": precision,
            "coverage": coverage,
            "source_diversity": diversity,
            "relevance_distribution": relevance_dist,
        },
        "efficiency_metrics": efficiency,
        "baseline_comparison": comparison,
    }