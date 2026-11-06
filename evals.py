"""
evals.py — FitDesi Baseline Evaluation Script (Phase 2)
========================================================
Runs a fixed set of tricky user queries through the current agent and saves
the raw responses to eval_results_v1.md for manual review.

Run this BEFORE any architectural changes (Advanced RAG, Multi-Agent) so you
have an objective baseline to compare against.

Usage:
    python evals.py
"""

import os
import uuid
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(override=True)

# Fake user context injected into every eval query
EVAL_USER_ID   = "eval_test_user_01"
EVAL_CONTEXT   = f"uid={EVAL_USER_ID}, date={datetime.today().date()}, diet=eggetarian, wt=70kg, goal=cut, cal=1800, pro=140g"

# ─── Test Cases ────────────────────────────────────────────────────────────────
# Each case has:
#   name   → short label for the report
#   query  → the user message
#   expect → what we WANT to see (used as a checklist comment — not auto-graded)
TEST_CASES = [
    {
        "name": "Mess Dal Logging + Mess Reality",
        "query": "I ate 2 katori mess dal and 3 roti",
        "expect": "Must use 2-3g protein per katori for mess dal (not 6-8g). Must log. Must NOT add ghee to roti unless user mentioned it.",
    },
    {
        "name": "3-Tier Pre-workout Suggestion",
        "query": "Suggest me a pre-workout meal",
        "expect": "Must give all 3 tiers (Mess/Grocery/Premium). Must NOT log. Must respect eggetarian. Must call get_food_macros before replying.",
    },
    {
        "name": "Ambiguous Input Guard",
        "query": "2 bananas",
        "expect": "Must ask 'Khaaya ya sirf check karna tha?' — must NOT auto-log or auto-calculate.",
    },
    {
        "name": "Golden Fat Rule — Omelette",
        "query": "I had a 2 egg omelette for breakfast",
        "expect": "Must add ~45 kcal cooking oil/butter card. Must log with fat included.",
    },
    {
        "name": "Science + Profile Personalization",
        "query": "Is creatine worth it for me?",
        "expect": "Must call get_science_facts. Must reference user's weight (70kg) and goal (cut) from context.",
    },
]

# ─── Run Evals ─────────────────────────────────────────────────────────────────
def run_evals():
    # Import agent here to avoid triggering LLM init at top of file
    from agent import agent_executor

    output_lines = [
        f"# FitDesi Baseline Eval Results — v1",
        f"**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Agent Config:** Single-agent, OpenRouter Llama-3.3-70b, PostgreSQL",
        f"**User Context:** {EVAL_CONTEXT}",
        f"\n---\n",
    ]

    print(f"\n🧪 Running {len(TEST_CASES)} eval cases...\n")

    for i, case in enumerate(TEST_CASES, 1):
        thread_id = f"eval_thread_{uuid.uuid4().hex[:8]}"
        config    = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
        inputs    = {
            "messages": [("user", case["query"])],
            "context":  EVAL_CONTEXT,
            "sender":   "",  # Phase 4: initialised empty, Manager node sets this
        }

        print(f"[{i}/{len(TEST_CASES)}] Running: {case['name']}...")

        try:
            result  = agent_executor.invoke(inputs, config=config)
            reply   = result["messages"][-1].content
            status  = "✅"
        except Exception as e:
            reply  = f"ERROR: {e}"
            status = "❌"

        print(f"  {status} Done.")

        # Rate limit guard for free-tier OpenRouter
        if i < len(TEST_CASES):
            wait = 15
            print(f"  ⏳ Waiting {wait}s to avoid rate limit...\n")
            time.sleep(wait)

        output_lines += [
            f"## {i}. {case['name']} {status}",
            f"**Query:** `{case['query']}`",
            f"\n**Expected Behaviour:**",
            f"> {case['expect']}",
            f"\n**Agent Response:**",
            f"```",
            reply.strip(),
            f"```",
            f"\n**Manual Grade:** [ ] Pass  [ ] Partial  [ ] Fail",
            f"\n---\n",
        ]

    output_path = "./eval_results_v1.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"✅ Eval complete! Results saved to: {output_path}")
    print("Open eval_results_v1.md and manually review each response.\n")


if __name__ == "__main__":
    run_evals()
