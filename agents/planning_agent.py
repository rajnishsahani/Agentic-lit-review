import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.shared_state import AgentState

combined_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research planning assistant.
    Given a research topic and themes from existing literature:
    1. First identify 3-5 research gaps or open questions
    2. Then write a structured research plan with proposed questions, methodology, and evaluation strategy
    3. Then create a READING ORDER — rank the papers from "read first" to "read last" so a beginner can build understanding progressively. Consider:
       - Foundational/survey papers come first
       - Papers that introduce key concepts come before papers that build on them
       - Older seminal works before newer extensions
       - General papers before specialized ones
    
    Format your response EXACTLY like this:
    GAPS:
    ["gap 1", "gap 2", "gap 3"]
    PLAN:
    [your research plan here]
    READING_ORDER:
    [
      {{"position": 1, "title": "exact paper title", "reason": "why read this first"}},
      {{"position": 2, "title": "exact paper title", "reason": "why read this second"}},
      ...
    ]"""),
    ("human", """Research topic: {topic}

Themes in existing literature:
{themes}

Paper summaries:
{summaries}

Full paper list with details:
{paper_details}

Provide gaps, research plan, and reading order:""")
])

def planning_agent(state: AgentState) -> AgentState:
    """Identifies research gaps, generates plan, and recommends reading order."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    print(f"\n[Planning Agent] Generating research plan and reading order...")
    
    # Build detailed paper info for reading order analysis
    paper_details = []
    for i, paper in enumerate(state.get("screened_papers", []), 1):
        detail = (
            f"{i}. Title: {paper['title']}\n"
            f"   Year: {paper.get('year', 'N/A')}\n"
            f"   Relevance: {paper.get('relevance_score', 'N/A')}\n"
            f"   Summary: {paper.get('summary', paper.get('abstract', '')[:200])}"
        )
        paper_details.append(detail)
    
    time.sleep(3)
    chain = combined_prompt | llm
    response = chain.invoke({
        "topic": state["research_topic"],
        "themes": "\n".join(state["themes"]),
        "summaries": "\n".join(state.get("summaries", [])[:10]),
        "paper_details": "\n\n".join(paper_details)
    })
    content = response.content
    
    # Parse gaps
    gaps = ["Further research needed"]
    plan_section = content
    reading_order = []
    
    if "GAPS:" in content and "PLAN:" in content:
        gaps_section = content.split("PLAN:")[0].replace("GAPS:", "").strip()
        try:
            gaps = eval(gaps_section.strip())
        except:
            gaps = ["Further research needed"]
        
        if "READING_ORDER:" in content:
            plan_section = content.split("READING_ORDER:")[0].split("PLAN:")[1].strip()
            reading_order_section = content.split("READING_ORDER:")[1].strip()
            try:
                reading_order = eval(reading_order_section.strip())
            except:
                reading_order = []
        else:
            plan_section = content.split("PLAN:")[1].strip()
    
    print(f"[Planning Agent] Research plan generated")
    print(f"[Planning Agent] Reading order: {len(reading_order)} papers sequenced")
    
    return {
        **state,
        "research_gaps": gaps,
        "research_plan": plan_section,
        "reading_order": reading_order,
        "messages": state["messages"] + ["Planning complete: research plan and reading order generated"]
    }