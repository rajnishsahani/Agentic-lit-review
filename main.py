import os
from dotenv import load_dotenv, find_dotenv
from controller.orchestrator import run_literature_review
from evaluation.metrics import generate_report, print_report

load_dotenv(find_dotenv())

def print_results(state):
    print("\n" + "="*60)
    print("LITERATURE REVIEW COMPLETE")
    print("="*60)
    
    print(f"\n📚 Research Topic: {state['research_topic']}")
    print(f"🔍 Search Queries Used: {len(state['search_queries'])}")
    print(f"📄 Papers Found: {len(state['raw_papers'])}")
    print(f"✅ Papers After Screening: {len(state['screened_papers'])}")
    
    print("\n--- TOP PAPERS ---")
    for i, paper in enumerate(state['screened_papers'][:5], 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Year: {paper['year']}")
        print(f"   Relevance: {paper['relevance_score']:.2f}")
        print(f"   Summary: {paper['summary']}")
        print(f"   URL: {paper['url']}")
    
    print("\n--- THEMES IDENTIFIED ---")
    for theme in state['themes']:
        print(f"  • {theme}")
    
    print("\n--- RESEARCH GAPS ---")
    for gap in state['research_gaps']:
        print(f"  • {gap}")
    
    print("\n--- RESEARCH PLAN ---")
    print(state['research_plan'])
    
    print("\n--- AGENT MESSAGES ---")
    for msg in state['messages']:
        print(f"  ✓ {msg}")

if __name__ == "__main__":
    topic = input("Enter your research topic: ")
    print(f"\nStarting literature review for: {topic}")
    print("This may take a minute...\n")
    
    result = run_literature_review(topic, max_iterations=2)
    print_results(result)
    
    # Generate and print evaluation report
    report = generate_report(result)
    print_report(report)