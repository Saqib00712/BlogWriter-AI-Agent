# BlogWriter AI Agent
A production-ready multi-agent system that autonomously researches and writes high-quality technical blog posts — using LangGraph for workflow orchestration, OpenAI GPT-4o-mini for generation, Tavily for real-time research, and parallel worker agents for fast section writing.

---

## What This Project Does

Most AI writing tools generate a single pass of text with no research and no structure. This system goes further by running a **full multi-agent pipeline** — a Router decides if research is needed, a Research Agent searches the web, an Orchestrator plans the structure, parallel Workers write each section simultaneously, and a Reducer assembles the final post with images and citations.

```
User provides topic
        ↓
Router Agent — decides research mode
        ├── closed_book  — no research needed
        ├── hybrid       — some research helpful
        └── open_book    — full research required
        ↓
Research Agent — Tavily web search + source synthesis
        ↓
Orchestrator Agent — creates 5-9 section blog plan
        ↓
Worker Agents (parallel) — write each section simultaneously
        ↓
Reducer Agent — merges sections + fetches images
        ↓
Final blog saved as .md file with citations and images
```

---

## Demo

**Input topic:** `Self Attention in Transformer Architecture`

```
🧭 Router: mode=hybrid, needs_research=True
🔍 Research: 12 unique evidence items found
📝 Orchestrator: 7 sections planned
📤 Fanning out to 7 parallel workers
✍️  Writing: Introduction to Self-Attention (200 words)
✍️  Writing: How Attention Scores Work (300 words)
✍️  Writing: Multi-Head Attention (250 words)
...
🖼️ Images: 2 fetched from Unsplash
💾 Saved: Understanding_Self-Attention_in_Transformers.md
✅ BLOG GENERATION COMPLETE
```

Total time: significantly faster than sequential writing because all sections run in parallel.

---

## Agent Workflow

### Agent 1 — Router
Analyzes the topic and decides the research strategy:
```python
mode: "closed_book"  # LLM knowledge only — no search needed
mode: "hybrid"       # Mix of LLM knowledge + some web research
mode: "open_book"    # Full web research required before writing
```

### Agent 2 — Research Agent
```python
# Searches web with Tavily, synthesizes evidence
queries = ["self attention transformer tutorial", "attention mechanism explained"]
evidence = tavily.search(queries)  # Returns 24 raw → 12 unique sources
```

### Agent 3 — Orchestrator
Creates a structured plan with goals, bullet points, word counts, and flags:
```python
class Task(BaseModel):
    id: int
    title: str
    goal: str
    bullets: List[str]
    target_words: int
    requires_research: bool
    requires_citations: bool
    requires_code: bool
```

### Agent 4 — Worker Agents (Parallel)
Each worker receives one task and writes its section independently — all run simultaneously using LangGraph's fanout pattern.

### Agent 5 — Reducer Agent
Merges all sections in correct order, fetches images from Unsplash/Picsum, and saves the final markdown file.

---

## Architecture

```
Main Graph
│
├── Router Node
│       ↓
├── Research Subgraph
│   ├── Search Node      — Tavily queries
│   └── Synthesize Node  — merge + deduplicate evidence
│       ↓
├── Orchestrator Node    — blog plan creation
│       ↓
├── Worker Nodes (Parallel Fanout)
│   ├── Worker 1 — Section 1
│   ├── Worker 2 — Section 2
│   └── Worker N — Section N
│       ↓
└── Reducer Subgraph
    ├── Merge Node           — combine all sections
    ├── Image Decision Node  — decide which sections need images
    └── Image Generation Node — fetch from Unsplash / Picsum
            ↓
    Final .md file saved to disk
    PostgreSQL checkpoint saves workflow state
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2.2-green?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-black?style=flat-square)
![Tavily](https://img.shields.io/badge/Tavily-Web%20Search-orange?style=flat-square)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Checkpointing-blue?style=flat-square)

- **LangGraph 1.2.2** — multi-agent workflow orchestration with subgraphs and parallel fanout
- **LangChain 1.3.2** — LLM framework and tool integration
- **OpenAI GPT-4o-mini** — text generation for all agents
- **Tavily** — real-time web research and source retrieval
- **PostgreSQL** — workflow state checkpointing and persistence
- **Unsplash / Picsum** — free image fetching for blog posts
- **LangSmith** — optional agent observability and tracing
- **Python 3.11+** — core runtime

---

## Project Structure

```
BlogWriter-AI-Agent/
│
├── notebooks/
│   └── blog_agent_ultimate.py    # Main agent — all 5 agents + graph
├── app.py                        # Flask web interface
├── backend.py                    # Backend logic
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
├── images/                       # Generated blog images
├── requirements.txt
├── .env.example
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Saqib00712/BlogWriter-AI-Agent.git
cd BlogWriter-AI-Agent
```

### 2. Create environment
```bash
conda create -n blogwriter python=3.11
conda activate blogwriter
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGSMITH_API_KEY=your_langsmith_key        # optional
DATABASE_URL=postgresql://user:pass@localhost:5432/blog_db  # optional
```

### 5. Run the agent
```bash
python notebooks/blog_agent_ultimate.py
```

### 6. Enter your topic
```
Enter blog topic: Designing a Production-Ready RAG System
```

The agent will research, plan, write, add images, and save a `.md` file automatically.

---

## Output

The agent generates 3 outputs:

| Output | Location | Description |
|--------|----------|-------------|
| Blog post | `{Blog_Title}.md` | Full markdown blog with all sections |
| Images | `images/` folder | Fetched from Unsplash/Picsum |
| Console logs | Terminal | Real-time progress for every agent step |

---

## Key Concepts Covered

- **LangGraph subgraphs** — modular agent design with Research and Reducer as independent subgraphs
- **Parallel fanout pattern** — all worker agents run simultaneously using LangGraph's map-reduce
- **Router agent** — LLM-based decision making to select research strategy before writing
- **Structured output with Pydantic** — `Plan`, `Task`, `EvidenceItem` models enforce consistent agent output
- **PostgreSQL checkpointing** — workflow state saved so long jobs can resume after failure
- **Evidence synthesis** — deduplicating and ranking 24 raw search results into 12 clean citations
- **Image decision agent** — LLM decides which sections need images and what to search for
- **LangSmith observability** — tracing every agent step with timing and token usage

---

## Supported Models

| Provider | Model | Use Case |
|----------|-------|----------|
| OpenAI | gpt-4o-mini | Default — fast and cost-efficient |
| OpenAI | gpt-4o | Premium — best quality output |
| Groq | llama-3.1-70b-versatile | Free fallback option |

---

## Related Certifications

Built applying skills from the IBM **Building AI Agents and Agentic Workflows Specialization** and **Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI** on Coursera.

[![IBM Badge](https://img.shields.io/badge/IBM-AI%20Agents%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)
[![IBM Badge](https://img.shields.io/badge/IBM-LangGraph%20%26%20Agentic%20AI-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)

---

## Author

**Muhammad Saqib**
- GitHub: [@Saqib00712](https://github.com/Saqib00712)
- LinkedIn: [muhammad-saqib](https://www.linkedin.com/in/muhammad-saqib-68b9b3374/)
- Email: saqibkhosa649@gmail.com
- Credly: [15x IBM Certified](https://www.credly.com/users/muhammad-saqib.361f9b8c)
