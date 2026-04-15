import os
import time
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from controller.orchestrator import run_literature_review
from evaluation.metrics import evaluate_run


def print_results(state):
    print("\n" + "=" * 60)
    print("LITERATURE REVIEW COMPLETE")
    print("=" * 60)
    
    print(f"\nResearch Topic: {state['research_topic']}")
    print(f"Search Queries Used: {len(state['search_queries'])}")
    print(f"Papers Found: {len(state['raw_papers'])}")
    print(f"Papers After Screening: {len(state['screened_papers'])}")
    
    print("\n--- TOP PAPERS ---")
    for i, paper in enumerate(state['screened_papers'][:5], 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Year: {paper.get('year', 'N/A')}")
        print(f"   Relevance: {paper.get('relevance_score', 0):.2f}")
        print(f"   Summary: {paper.get('summary', 'N/A')}")
    
    print("\n--- THEMES IDENTIFIED ---")
    for theme in state['themes']:
        print(f"  - {theme}")
    
    print("\n--- RESEARCH GAPS ---")
    for gap in state['research_gaps']:
        print(f"  - {gap}")
    
    print("\n--- RESEARCH PLAN ---")
    print(state['research_plan'])

    print("\n--- RECOMMENDED READING ORDER ---")
    for entry in state.get("reading_order", []):
        print(f"\n  {entry.get('position', '?')}. {entry.get('title', 'Unknown')}")
        print(f"     Why: {entry.get('reason', 'N/A')}")
    
    print("\n--- AGENT MESSAGES ---")
    for msg in state['messages']:
        print(f"  > {msg}")


if __name__ == "__main__":
    topic = input("Enter your research topic: ")
    print(f"\nStarting literature review for: {topic}")
    print("This may take a few minutes (rate limit delays between agents)...\n")
    
    # Time the pipeline
    start_time = time.time()
    result = run_literature_review(topic, max_iterations=2)
    execution_time = time.time() - start_time
    
    # Print pipeline results
    print_results(result)
    
    # Run evaluation metrics
    evaluation = evaluate_run(result, execution_time, topic)
    
    print(f"\nTotal pipeline time: {execution_time:.1f}s")
    print("Done!")