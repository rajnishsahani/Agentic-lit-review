# Agentic Literature Review System
### CIS 600 — Applied Agentic AI Systems | Syracuse University | Spring 2026

> An autonomous multi-agent system that conducts academic literature reviews and generates structured research plans using LangGraph, Google Gemini, and the arXiv API.

---

## Team

| Name | NetID | Role |
|---|---|---|
| Rajnish Sahani | rsahani@syr.edu | Search Agent + Orchestrator |
| Deven Wagh | dpwagh@syr.edu | Screening Agent |
| Hangye Li | hli299@syr.edu | Synthesis Agent |
| Yonghao Li | yli598@syr.edu | Planning Agent |

---

## What It Does

Given a research topic, the system autonomously:

1. **Searches** arXiv for relevant papers using LLM-generated queries
2. **Screens** papers for relevance using AI scoring
3. **Synthesizes** summaries and identifies themes across papers
4. **Plans** research directions by identifying gaps in existing literature

All coordinated by a LangGraph orchestrator with an adaptive feedback loop.

---

## System Architecture

```
Research Topic
      │
      ▼
┌─────────────┐
│ Search Agent │  ← Generates queries, searches arXiv
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Screening Agent   │  ← Scores relevance (0.0–1.0), filters papers
└────────┬─────────┘
         │
    [< 3 papers?] ──── Yes ──→ Loop back to Search
         │ No
         ▼
┌──────────────────┐
│ Synthesis Agent   │  ← Summarizes papers, identifies themes
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Planning Agent   │  ← Identifies gaps, generates research plan
└────────┬────────┘
         │
         ▼
    Final Output
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM | Google Gemini 2.0 Flash |
| Paper Retrieval | arXiv API |
| Framework | LangChain |
| Language | Python 3.11 |

---

## Project Structure

```
Agentic-lit-review/
├── agents/
│   ├── search_agent.py       # Query generation + arXiv retrieval
│   ├── screening_agent.py    # Relevance scoring + filtering
│   ├── synthesis_agent.py    # Summarization + theme identification
│   └── planning_agent.py     # Gap analysis + research planning
├── controller/
│   └── orchestrator.py       # LangGraph graph + feedback loop
├── tools/
│   └── arxiv_tool.py         # arXiv API wrapper
├── state/
│   └── shared_state.py       # TypedDict state schema
├── evaluation/
│   └── metrics.py            # Coverage, relevance, efficiency metrics
├── notebooks/
│   └── demo.ipynb            # End-to-end demonstration
├── main.py                   # Entry point
├── .env.example              # API key template
└── requirements.txt          # Dependencies
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/rajnishsahani/Agentic-lit-review.git
cd Agentic-lit-review
```

**2. Create and activate virtual environment**
```bash
python3.11 -m venv venv
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install langchain langgraph langchain-google-genai langchain-core arxiv requests python-dotenv
```

**4. Configure API key**
```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key:
# GOOGLE_API_KEY=your-key-here
```

**5. Run the system**
```bash
python main.py
```

---

## Example Output

```
Enter your research topic: agentic AI systems for literature review

[Search Agent] Searching for: agentic AI systems for literature review
[Search Agent] Generated queries: ['agentic AI autonomous research', ...]
[Search Agent] Found 14 unique papers

[Screening Agent] Screening 14 papers...
[Screening Agent] 9 papers passed screening

[Synthesis Agent] Synthesizing 9 papers...
[Synthesis Agent] Identified themes: ['Multi-agent coordination', 'RAG pipelines', ...]

[Planning Agent] Generating research plan...

============================================================
LITERATURE REVIEW COMPLETE
============================================================

📚 Research Topic: agentic AI systems for literature review
🔍 Search Queries Used: 3
📄 Papers Found: 14
✅ Papers After Screening: 9

--- TOP PAPERS ---
1. Autonomous Agents for Scientific Discovery
   Year: 2024 | Relevance: 0.95
   Summary: ...

--- THEMES IDENTIFIED ---
  • Multi-agent coordination for complex tasks
  • RAG-based knowledge retrieval
  • ...

--- RESEARCH GAPS ---
  • Lack of evaluation benchmarks for agentic review systems
  • ...

--- RESEARCH PLAN ---
  1. Proposed Research Questions: ...
  2. Suggested Methodology: ...
  3. Evaluation Strategy: ...
```

---

## References

1. Yao et al., ReAct, arXiv, 2022
2. Park et al., Generative Agents, arXiv, 2023
3. Shinn et al., Reflexion, arXiv, 2023
4. Wooldridge, An Introduction to Multi-Agent Systems, Wiley, 2009
5. LangGraph Documentation — https://langchain-ai.github.io/langgraph/