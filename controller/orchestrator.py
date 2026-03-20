from langgraph.graph import StateGraph, END
from state.shared_state import AgentState
from agents.search_agent import search_agent
from agents.screening_agent import screening_agent
from agents.synthesis_agent import synthesis_agent
from agents.planning_agent import planning_agent

def should_continue(state: AgentState) -> str:
    """Decide whether to continue or end the workflow."""
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    if len(state["screened_papers"]) < 3 and state["iteration"] < state["max_iterations"]:
        return "search_again"
    return "end"

def increment_iteration(state: AgentState) -> AgentState:
    """Increment the iteration counter."""
    return {**state, "iteration": state["iteration"] + 1}

def build_graph():
    """Build and return the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("search", search_agent)
    workflow.add_node("screen", screening_agent)
    workflow.add_node("synthesize", synthesis_agent)
    workflow.add_node("plan", planning_agent)
    workflow.add_node("increment", increment_iteration)
    
    # Add edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "screen")
    workflow.add_edge("screen", "increment")
    
    # Conditional edge — loop back or continue
    workflow.add_conditional_edges(
        "increment",
        should_continue,
        {
            "search_again": "search",
            "end": "synthesize"
        }
    )
    
    workflow.add_edge("synthesize", "plan")
    workflow.add_edge("plan", END)
    
    return workflow.compile()

def run_literature_review(topic: str, max_iterations: int = 2):
    """Run the full literature review pipeline."""
    
    initial_state: AgentState = {
        "research_topic": topic,
        "search_queries": [],
        "raw_papers": [],
        "screened_papers": [],
        "summaries": [],
        "themes": [],
        "research_gaps": [],
        "research_plan": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "status": "running",
        "messages": []
    }
    
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    
    return final_state