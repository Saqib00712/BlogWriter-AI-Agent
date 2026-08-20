#!/usr/bin/env python
# coding: utf-8

"""
ULTIMATE BLOG WRITER AGENT - Standalone Version
Features:
- Router (decides if research is needed)
- Tavily Search (web research)
- OpenAI GPT-4o-mini for text generation
- DALL-E 2/3 for image generation (auto-fallback)
- Evidence-based writing with citations
- Smart planning with flags
- Automatic image placeholders and generation
- PostgreSQL checkpointing (with memory fallback)
"""

from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

# =========================
# Database Configuration
# =========================

def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env")
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    return database_url

# =========================
# Schemas / Models
# =========================

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should be able to do/understand after this section.")
    bullets: List[str] = Field(..., min_length=3, max_length=6, description="3–6 concrete, non-overlapping subpoints.")
    target_words: int = Field(..., description="Target word count for this section (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str

# =========================
# Initialize LLM - OPENAI + GROQ FALLBACK
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if OPENAI_API_KEY:
    print("=" * 60)
    print("🤖 ULTIMATE BLOG WRITER AGENT")
    print("=" * 60)
    print("✅ Using OpenAI LLM")
    print("   - Text Model: gpt-4o-mini")
    print("   - Image Model: DALL-E 2/3 (auto-fallback)")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0,
    )
    IMAGE_GENERATION_AVAILABLE = True
    
elif GROQ_API_KEY:
    print("=" * 60)
    print("🤖 ULTIMATE BLOG WRITER AGENT")
    print("=" * 60)
    print("✅ Using Groq LLM (fallback)")
    print("   - Text Model: llama-3.1-70b-versatile")
    print("   - Image Model: Not available")
    print("=" * 60)
    
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0,
    )
    IMAGE_GENERATION_AVAILABLE = False
    
else:
    raise ValueError(
        "No API key found.\n"
        "Please set in .env:\n"
        "  - OPENAI_API_KEY (recommended)\n"
        "  - GROQ_API_KEY (fallback)"
    )

# =========================
# Image Generation - DALL-E 2/3 Auto-Fallback
# =========================

def generate_image_dalle(prompt: str, size: str = "1024x1024", quality: str = "standard") -> Optional[bytes]:
    """
    Generate image using DALL-E (OpenAI)
    Tries DALL-E 3 first, falls back to DALL-E 2 if needed
    Requires: pip install openai requests Pillow
    Env var: OPENAI_API_KEY
    """
    if not IMAGE_GENERATION_AVAILABLE:
        print("   ⚠️ Image generation not available (no OpenAI API key)")
        return None
    
    try:
        from openai import OpenAI
        import requests
    except ImportError as e:
        print(f"   ⚠️ Missing dependency: {e}")
        print("   Run: pip install openai requests")
        return None
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("   ⚠️ OPENAI_API_KEY is not set")
        return None
    
    client = OpenAI(api_key=api_key)
    
    print(f"   🎨 Generating DALL-E image: {prompt[:50]}...")
    
    # Try DALL-E 3 first, fallback to DALL-E 2
    models_to_try = ["dall-e-3", "dall-e-2"]
    
    # DALL-E 2 only supports specific sizes
    dalle2_sizes = ["1024x1024", "512x512", "256x256"]
    dalle3_sizes = ["1024x1024", "1024x1536", "1536x1024"]
    
    for model in models_to_try:
        try:
            # Adjust size for DALL-E 2
            if model == "dall-e-2" and size not in dalle2_sizes:
                size = "1024x1024"
                print(f"   📐 Using size {size} for DALL-E 2")
            
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                n=1,
            )
            image_url = response.data[0].url
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()
            print(f"   ✅ Generated with {model}")
            return img_response.content
        except Exception as e:
            error_msg = str(e)
            if "model 'dall-e-3' does not exist" in error_msg:
                print(f"   ⚠️ DALL-E 3 not available, trying DALL-E 2...")
                continue
            elif "billing" in error_msg.lower() or "quota" in error_msg.lower():
                print(f"   ⚠️ Billing/quota issue: {e}")
                print("   Please check your OpenAI account billing")
                return None
            else:
                print(f"   ⚠️ {model} error: {e}")
                continue
    
    print("   ⚠️ All image models failed")
    return None

# =========================
# Router Node
# =========================

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): Evergreen topics (concepts, fundamentals).
- hybrid (needs_research=true): Mostly evergreen but needs up-to-date examples/tools/models.
- open_book (needs_research=true): Volatile topics (weekly roundups, "latest", rankings, pricing).

If needs_research=true: Output 3–10 high-signal, specific queries.
"""

def router_node(state: State) -> dict:
    print(f"\n🧭 Routing for: {state['topic']}")
    
    topic = state["topic"]
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {topic}")
    ])
    
    print(f"   📊 Mode: {decision.mode}")
    print(f"   🔍 Needs research: {decision.needs_research}")
    if decision.needs_research:
        print(f"   🔎 Queries: {decision.queries}")
    
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# =========================
# Research Node (Tavily)
# =========================

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    
    normalized: List[dict] = []
    for r in results or []:
        normalized.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "snippet": r.get("content") or r.get("snippet") or "",
            "published_at": r.get("published_date") or r.get("published_at"),
            "source": r.get("source"),
        })
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- If published date present, keep as YYYY-MM-DD. If missing, set null.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    print(f"\n🔍 Researching topic...")
    
    queries = state.get("queries", []) or []
    raw_results: List[dict] = []
    
    for q in queries:
        print(f"   🔎 Searching: {q}")
        raw_results.extend(_tavily_search(q, max_results=6))
    
    if not raw_results:
        print("   ⚠️ No results found")
        return {"evidence": []}
    
    print(f"   📄 Found {len(raw_results)} raw results")
    
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw results:\n{raw_results}")
    ])
    
    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    
    print(f"   ✅ {len(dedup)} unique evidence items")
    return {"evidence": list(dedup.values())}

# =========================
# Orchestrator Node
# =========================

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.

Create a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks).
- Each task: goal (1 sentence), 3–6 concrete bullets, target words (120–550).
- Include flags: requires_research, requires_citations, requires_code.

Quality bar:
- Assume developer audience; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Include at least 2 of: code sketch, edge cases, performance, security, debugging tips.

Grounding rules:
- closed_book: evergreen only, no evidence needed.
- hybrid: use evidence for fresh examples, mark requires_research=True and requires_citations=True.
- open_book: set blog_kind="news_roundup", every section summarizes events + implications.
- If evidence insufficient, transparently say so.

Output must strictly match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    print(f"\n📝 Creating blog plan...")
    
    planner = llm.with_structured_output(Plan)
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")
    
    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\nMode: {mode}\n\n"
                f"Evidence (ONLY use for fresh claims; may be empty):\n"
                f"{[e.model_dump() for e in evidence][:16]}"
            )
        )
    ])
    
    print(f"✅ Plan created!")
    print(f"   📌 Title: {plan.blog_title}")
    print(f"   👥 Audience: {plan.audience}")
    print(f"   📝 Sections: {len(plan.tasks)}")
    print(f"   📂 Kind: {plan.blog_kind}")
    
    return {"plan": plan}

# =========================
# Fanout
# =========================

def fanout(state: State):
    print(f"\n📤 Fanning out to {len(state['plan'].tasks)} workers")
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
        })
        for task in state["plan"].tasks
    ]

# =========================
# Worker Node
# =========================

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.

Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the Goal and cover ALL Bullets in order.
- Stay close to Target words (±15%).
- Output ONLY the section content (no H1 title, no extra commentary).
- Start with '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": focus on summarizing events, NOT tutorials.

Grounding policy:
- open_book mode: ONLY use Evidence URLs for event claims, cite as ([Source](URL)).
- If requires_citations=True: cite Evidence URLs for outside-world claims.
- Evergreen reasoning OK without citations.

Code:
- If requires_code=True: include at least one minimal code snippet.

Style:
- Short paragraphs, bullets where helpful, code fences.
- Be precise and implementation-oriented.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    
    flags = []
    if task.requires_research: flags.append("🔬")
    if task.requires_citations: flags.append("📚")
    if task.requires_code: flags.append("💻")
    flag_str = " ".join(flags) if flags else ""
    
    print(f"\n✍️ Writing: {task.title} ({task.target_words} words) {flag_str}")

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
            for e in evidence[:20]
        )

    section_md = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(
            content=(
                f"Blog title: {plan.blog_title}\n"
                f"Audience: {plan.audience}\n"
                f"Tone: {plan.tone}\n"
                f"Blog kind: {plan.blog_kind}\n"
                f"Constraints: {plan.constraints}\n"
                f"Topic: {topic}\n"
                f"Mode: {mode}\n\n"
                f"Section title: {task.title}\n"
                f"Goal: {task.goal}\n"
                f"Target words: {task.target_words}\n"
                f"Tags: {task.tags}\n"
                f"requires_research: {task.requires_research}\n"
                f"requires_citations: {task.requires_citations}\n"
                f"requires_code: {task.requires_code}\n"
                f"Bullets:{bullets_text}\n\n"
                f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
            )
        )
    ]).content.strip()

    print(f"✅ Section completed: {task.title} ({len(section_md)} chars)")
    return {"sections": [(task.id, section_md)]}

# =========================
# Reducer with Images (Subgraph)
# =========================

def merge_content(state: State) -> dict:
    plan = state["plan"]
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    print(f"\n📝 Merged {len(ordered_sections)} sections")
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.

Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding.
- Insert placeholders: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders = input, images=[].
- Prefer technical diagrams.

Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    print(f"\n🖼️ Deciding images...")
    
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    
    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(
            content=(
                f"Blog kind: {plan.blog_kind}\n"
                f"Topic: {state['topic']}\n\n"
                "Insert placeholders + propose image prompts.\n\n"
                f"{merged_md}"
            )
        )
    ])
    
    print(f"   📸 Images requested: {len(image_plan.images)}")
    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def generate_and_place_images(state: State) -> dict:
    print(f"\n🎨 Generating images...")
    
    plan = state["plan"]
    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []
    
    if not image_specs:
        filename = f"{plan.blog_title}.md"
        Path(filename).write_text(md, encoding="utf-8")
        print(f"💾 Blog saved to: {filename}")
        return {"final": md}
    
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    
    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename
        
        if not out_path.exists():
            print(f"   🎨 Generating: {filename}")
            img_bytes = generate_image_dalle(
                spec["prompt"],
                spec.get("size", "1024x1024"),
                spec.get("quality", "standard")
            )
            if img_bytes:
                out_path.write_bytes(img_bytes)
                print(f"   ✅ Saved: {filename}")
            else:
                # Image generation failed - keep placeholder but add note
                prompt_block = (
                    f"> **[IMAGE NOT GENERATED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue
        
        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)
    
    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(md, encoding="utf-8")
    print(f"💾 Final blog saved to: {filename}")
    if image_specs:
        print(f"📁 Images saved in: images/")
    return {"final": md}

# =========================
# Build Reducer Subgraph
# =========================

reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# =========================
# Build Main Graph
# =========================

g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

# =========================
# PostgreSQL Checkpointer
# =========================

try:
    DATABASE_URL = get_database_url()
    print("\n🔗 Connecting to PostgreSQL...")
    _conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
    checkpointer = PostgresSaver(_conn)
    checkpointer.setup()
    print("✅ PostgreSQL checkpointer ready")
except Exception as e:
    print(f"⚠️ PostgreSQL not available, using memory checkpointer: {e}")
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()

app = g.compile(checkpointer=checkpointer)
print("\n✅ Graph compiled successfully")
print("=" * 60)
print("🚀 ULTIMATE BLOG WRITER AGENT READY")
print("=" * 60)

# =========================
# Runner Function
# =========================

def run(topic: str):
    """
    Run the blog agent with a given topic
    """
    print("\n" + "=" * 60)
    print("🚀 STARTING BLOG GENERATION")
    print("=" * 60)
    print(f"📌 Topic: {topic}")
    print("=" * 60)
    
    config = {
        "configurable": {
            "thread_id": f"blog_{date.today().isoformat()}"
        }
    }
    
    try:
        result = app.invoke(
            {
                "topic": topic,
                "mode": "",
                "needs_research": False,
                "queries": [],
                "evidence": [],
                "plan": None,
                "sections": [],
                "merged_md": "",
                "md_with_placeholders": "",
                "image_specs": [],
                "final": "",
            },
            config=config
        )
        
        print("\n" + "=" * 60)
        print("✅ BLOG GENERATION COMPLETE!")
        print("=" * 60)
        
        plan = result.get("plan")
        if plan:
            print(f"\n📊 Blog Summary:")
            print(f"   Title: {plan.blog_title}")
            print(f"   Audience: {plan.audience}")
            print(f"   Tone: {plan.tone}")
            print(f"   Kind: {plan.blog_kind}")
            print(f"   Sections: {len(plan.tasks)}")
            
            print("\n   Section Breakdown:")
            for task in plan.tasks:
                flags = []
                if task.requires_research: flags.append("🔬")
                if task.requires_citations: flags.append("📚")
                if task.requires_code: flags.append("💻")
                flag_str = " ".join(flags) if flags else ""
                print(f"      • {task.title} - {task.target_words} words {flag_str}")
        
        print("\n📄 Blog Preview:")
        print("-" * 60)
        final = result.get("final", "")
        preview = final[:600] + "..." if len(final) > 600 else final
        print(preview)
        print("-" * 60)
        
        filename = f"{plan.blog_title}.md" if plan else "blog.md"
        print(f"\n💾 Full blog saved to: {filename}")
        
        image_specs = result.get("image_specs", [])
        if image_specs:
            print(f"🖼️ Images generated: {len(image_specs)}")
            for img in image_specs:
                print(f"   • {img.get('filename')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

# =========================
# Main Entry Point
# =========================

if __name__ == "__main__":
    topic = input("\n📝 Enter blog topic: ").strip()
    if not topic:
        topic = "Self Attention in Transformer Architecture"
    run(topic)