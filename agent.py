import os
import uuid
import re
import operator
import typing
from datetime import date, timedelta
from dotenv import load_dotenv
from typing import Annotated, Sequence, Literal

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func

from database import SessionLocal
import models
from models import FoodItem, RagDocument

# ── MCP: OpenFoodFacts Fallback ──────────────────────────────────────────────
import sys
import asyncio
import httpx as _httpx
from pathlib import Path

load_dotenv()

# ==========================================
# 0. MCP: OpenFoodFacts Fallback Helper
# ==========================================

_OFF_ENDPOINTS = [
    "https://in.openfoodfacts.org/cgi/search.pl",
    "https://world.openfoodfacts.org/cgi/search.pl",
]
_OFF_HEADERS = {
    "User-Agent": "FitDesi-AI/1.0 (educational; github.com/AdityaAryan01)"
}

def _query_off_sync(query: str) -> str:
    """
    Synchronous wrapper around the OpenFoodFacts REST API.
    Called inside get_food_macros when local Postgres returns NOT FOUND.
    """
    params = {
        "search_terms": query,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 5,
        "fields": "product_name,nutriments,serving_size,brands",
        "lc": "en",
    }
    data = None
    for url in _OFF_ENDPOINTS:
        try:
            resp = _httpx.get(url, params=params, headers=_OFF_HEADERS, timeout=5.0)
            if resp.status_code == 200 and "json" in resp.headers.get("content-type", ""):
                data = resp.json()
                break
        except Exception:
            continue

    if not data:
        return "NOT_FOUND_IN_OFF"

    for product in data.get("products", []):
        name   = (product.get("product_name") or "").strip()
        brand  = (product.get("brands") or "").strip()
        n      = product.get("nutriments", {})
        kcal   = n.get("energy-kcal_100g") or n.get("energy-kcal") or n.get("energy_100g")
        prot   = n.get("proteins_100g")    or n.get("proteins")
        carbs  = n.get("carbohydrates_100g") or n.get("carbohydrates")
        fat    = n.get("fat_100g")          or n.get("fat")

        if not name or (kcal is None and prot is None):
            continue

        display = f"{brand} {name}".strip() if brand else name
        serving = product.get("serving_size") or "100g"
        return (
            f"[OpenFoodFacts] {display}: "
            f"{round(float(kcal or 0), 1)} kcal | "
            f"{round(float(prot or 0), 1)}g protein | "
            f"{round(float(carbs or 0), 1)}g carbs | "
            f"{round(float(fat or 0), 1)}g fat "
            f"per 100g (label serving: {serving})"
        )

    return "NOT_FOUND_IN_OFF"


# ==========================================
# 1. EMBEDDING MODEL (shared)
# ==========================================
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# ==========================================
# 2. LLM DEFINITIONS
# ==========================================

# Heavy lifter - reasoning, math, tool-calling
llm_70b = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=1000,
)

# Fast brain - routing and formatting (cheap)
llm_8b = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=500,
)

# ==========================================
# 3. TOOL SCHEMAS (Pydantic)
# ==========================================

class FoodMacroInput(BaseModel):
    food_names: list[str] = Field(
        ...,
        description="A list of SINGLE food items to search for (e.g., ['egg', 'bread']). NEVER put combined meals into a single string."
    )

class ScienceQueryInput(BaseModel):
    query: str = Field(
        ...,
        description="The scientific question, fitness myth, or supplement to verify."
    )

class MealHistoryInput(BaseModel):
    user_id: str = Field(..., description="The Firebase UID of the user")
    days: int = Field(default=1, description="How many days back to check (1=today, 7=last week)")

    @field_validator('days', mode='before')
    @classmethod
    def coerce_days_to_int(cls, v):
        return int(v)

class MealLogInput(BaseModel):
    user_id: str = Field(..., description="The Firebase UID of the user")
    food_name: str = Field(..., description="Name of the food eaten")
    calories: str = Field(..., description="Total calories as a string, e.g. '200'")
    protein: str = Field(..., description="Total protein as a string, e.g. '12'")

class UserProfileInput(BaseModel):
    user_id: str = Field(..., description="The Firebase UID of the user")

# ==========================================
# 4. TOOLS
# ==========================================

@tool(args_schema=FoodMacroInput)
def get_food_macros(food_names: list[str]) -> str:
    """Search the food database for macros using fuzzy search (pg_trgm).
    Results reflect homestyle Indian/mess cooking.
    CRITICAL RULES:
    1. NEVER pass combined meals or long phrases in a single list element.
    2. Pass each atomic ingredient separately.
    3. If NOT FOUND: estimate using mess-style assumptions only.
    """
    db = SessionLocal()
    try:
        final_results = []
        for food_name in food_names:
            search_term = food_name.strip().lower()

            # Phase 3 Fix: pg_trgm fuzzy similarity search
            try:
                results = (
                    db.query(FoodItem)
                    .filter(
                        func.similarity(func.lower(FoodItem.item_name), search_term) > 0.15
                    )
                    .order_by(
                        func.similarity(func.lower(FoodItem.item_name), search_term).desc()
                    )
                    .limit(4)
                    .all()
                )
            except Exception:
                # Fallback to ilike if pg_trgm not available
                results = (
                    db.query(FoodItem)
                    .filter(FoodItem.item_name.ilike(f"%{search_term}%"))
                    .limit(4)
                    .all()
                )

            # Word-level fallback if trigram found nothing
            if not results:
                words = search_term.split()
                for word in reversed(words):
                    if len(word) > 3:
                        fallback = (
                            db.query(FoodItem)
                            .filter(FoodItem.item_name.ilike(f"%{word}%"))
                            .limit(3)
                            .all()
                        )
                        if fallback:
                            results = fallback
                            break

            if results:
                lines = [
                    f"- {f.item_name}: {f.calories} kcal, {f.protein}g protein, "
                    f"{f.carbs}g carbs, {f.fat}g fat per {f.serving_size}"
                    for f in results
                ]
                final_results.append(f"Matches for '{food_name}':\n" + "\n".join(lines))
            else:
                # ── MCP Fallback: try OpenFoodFacts before giving up ─────────
                print(f"[OFF Fallback] Local DB missed '{food_name}', querying OpenFoodFacts...", flush=True)
                off_result = _query_off_sync(food_name)
                if off_result.startswith("[OpenFoodFacts]"):
                    print(f"[OFF Fallback] Hit: {off_result[:80]}", flush=True)
                    final_results.append(off_result)
                else:
                    # OFF also couldn't find it — fall back to mess-style estimate
                    print(f"[OFF Fallback] Miss ({off_result}), using mess-estimate.", flush=True)
                    final_results.append(
                        f"'{food_name}' NOT FOUND in local DB or OpenFoodFacts. "
                        f"Estimate using mess-style rules: "
                        f"dal ~6-8g protein/100g, sabzi ~50 kcal/100g, oil 1-2 tsp."
                    )

        return "\n\n".join(final_results)
    except Exception as e:
        return f"Database error in get_food_macros: {str(e)}"
    finally:
        db.close()


@tool(args_schema=ScienceQueryInput)
def get_science_facts(query: str) -> str:
    """Use this to verify scientific fitness facts, supplements, myths, or ICMR guidelines.
    Searches the knowledge base of Indian diet and fitness research papers.
    """
    db = SessionLocal()
    try:
        # Phase 3 Fix: pgvector cosine distance search on RagDocument
        query_vector = embeddings_model.embed_query(query)

        results = (
            db.query(RagDocument)
            .order_by(RagDocument.embedding.cosine_distance(query_vector))
            .limit(3)
            .all()
        )

        if results:
            context = "\n\n".join([doc.content for doc in results])
            return f"Scientific Context retrieved:\n{context}"
        return "No scientific context found in the research papers for this query."
    except Exception as e:
        return f"RAG retrieval error: {str(e)}"
    finally:
        db.close()


@tool(args_schema=MealLogInput)
def log_meal_to_database(user_id: str, food_name: str, calories: str, protein: str) -> str:
    """Save a meal to the user's daily log AFTER macros have been calculated.
    ONLY call when the user EXPLICITLY confirmed they already ate something (past tense).
    NEVER call for hypothetical, suggestion, or future meals.
    """
    try:
        cal_val = float(calories)
        pro_val = float(protein)
    except ValueError:
        return "ERROR: Calories and protein must be numeric. Recalculate and try again."

    if cal_val < 0 or cal_val > 3000:
        return "SECURITY BLOCK: Calories exceed 3000 kcal for a single meal. Ask user to verify."
    if pro_val < 0 or pro_val > 300:
        return "ERROR: Invalid protein amount. Must be 0-300g."

    clean_food_name = food_name.replace(";", "").replace("--", "").strip()

    db = SessionLocal()
    try:
        new_log = models.DailyLog(
            user_id=user_id,
            date=date.today(),
            food_name=clean_food_name,
            calories=round(cal_val, 1),
            protein=round(pro_val, 1),
        )
        db.add(new_log)
        db.commit()
        return f"SUCCESS: Logged '{clean_food_name}' ({cal_val} kcal, {pro_val}g protein) for user {user_id}."
    except Exception as e:
        db.rollback()
        return f"DATABASE ERROR: Failed to log meal. Detail: {str(e)}"
    finally:
        db.close()


@tool(args_schema=MealHistoryInput)
def get_user_meal_history(user_id: str, days: int = 1) -> str:
    """Check what the user ate in the past.
    days=1 means today only. days=7 means last 7 days.
    """
    db = SessionLocal()
    try:
        start_date = date.today() - timedelta(days=days - 1)
        logs = (
            db.query(models.DailyLog)
            .filter(
                models.DailyLog.user_id == user_id,
                models.DailyLog.date >= start_date,
            )
            .order_by(models.DailyLog.date.desc())
            .all()
        )

        if not logs:
            return f"No meal logs found for user {user_id} in the past {days} day(s)."

        history = f"Meal logs for {user_id} (last {days} day(s)):\n"
        for log in logs:
            history += f"- {log.date}: {log.food_name} ({log.calories} kcal, {log.protein}g protein)\n"
        return history
    finally:
        db.close()


@tool(args_schema=UserProfileInput)
def get_user_profile(user_id: str) -> str:
    """Retrieve the user full fitness profile including goal, diet type, weight, and macro targets.
    Use this before giving any personalised advice or meal suggestions.
    """
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            return f"No profile found for user {user_id}. Ask them to complete their profile first."
        return (
            f"User Profile for {user_id}:\n"
            f"  Name: {user.name}\n"
            f"  Age: {user.age} | Gender: {user.gender}\n"
            f"  Weight: {user.weight_kg}kg | Height: {user.height_cm}cm\n"
            f"  Goal: {user.goal} | Diet: {user.diet_type}\n"
            f"  Activity Level: {user.activity_level}\n"
            f"  Daily Targets: {user.target_calories} kcal | {user.target_protein}g protein"
        )
    finally:
        db.close()


# ==========================================
# 5. TOOL BUNDLES PER AGENT
# ==========================================

# Calculator: read + write DB access
calculator_tools = [get_food_macros, log_meal_to_database, get_user_meal_history]

# Nutritionist: read-only macros + science + profile
nutritionist_tools = [get_food_macros, get_science_facts, get_user_profile]

# All tools registered in ToolNode
all_tools = [get_food_macros, log_meal_to_database, get_user_meal_history,
             get_science_facts, get_user_profile]

tool_node = ToolNode(all_tools)

# ==========================================
# 6. GRAPH STATE
# ==========================================

class AgentState(typing.TypedDict):
    messages: Annotated[Sequence[AnyMessage], operator.add]
    context: str   # user profile string injected by main.py
    sender: str    # tracks which specialist is active

# ==========================================
# 7. SYSTEM PROMPTS
# ==========================================

MANAGER_SYSTEM_PROMPT = """You are the FitDesi routing agent. Your ONLY job is to classify the user message and output exactly one word - nothing else.

Classification Rules:
- "calculator"   -> User is REPORTING they ate something (logging), asking what is IN a food (macro query), or asking to see past meal history.
- "nutritionist" -> User is asking for meal SUGGESTIONS, fitness science, supplement advice, pre/post-workout plans, diet tips, or anything requiring science-backed personalized advice.
- "general"      -> Greetings, off-topic chat, or simple yes/no conversational replies.

Examples:
"I ate 2 katori mess dal and 3 roti"  -> calculator
"Suggest me a pre-workout meal"       -> nutritionist
"How much protein in an egg?"         -> calculator
"Is creatine worth it for me?"        -> nutritionist
"Show me what I ate today"            -> calculator
"What is a good post-workout meal?"   -> nutritionist
"Hey bro what is up"                  -> general
"2 bananas"                           -> calculator

Output ONLY one of: calculator | nutritionist | general
No explanation. No punctuation. Just the single word."""

CALCULATOR_SYSTEM_PROMPT = """You are FitDesi Macro Calculator - a hyper-precise, no-nonsense math engine for Indian students. You handle meal logging and macro math only.

CURRENT USER CONTEXT (provided in messages) contains: user_id, date, goal, targets, weight, diet. Extract user_id from [CONTEXT] for all tool calls.

QUANTITY RULE - HIGHEST PRIORITY:
Always use the EXACT quantity the user stated. Read the number carefully.
- "3 eggs" -> calculate for 3 eggs. NEVER log 2 eggs.
- "2 pc bread" -> calculate for 2 slices. NEVER round down.
Before calling log_meal_to_database, re-read the user message and confirm: "Am I using their exact quantity?"

INTENT RULES (classify first, act second):
A) ATE IT (past tense: "khaaya", "had", "ate", "kha liya", "finished") -> get_food_macros per ingredient -> log_meal_to_database -> reply with macros + confirm
B) MACRO QUERY ("calories in", "protein of", "kitna protein", "is X healthy") -> get_food_macros -> reply only. NEVER log.
E) PAST MEALS ("aaj kya khaaya", "show logs") -> get_user_meal_history -> summarize
AMBIGUOUS (no verb, e.g. "2 eggs") -> ask "Khaaya ya sirf check karna tha?" DO NOT log or calculate.

SEARCH RULES:
- Generic Portions: "1 plate Chicken Biryani" -> search "Chicken Biryani", scale 100g macros to portion size.
- Detailed Breakdown: user explicitly lists ingredients -> search each ingredient separately.
- The Golden Fat Rule: For any tawa/pan-cooked dish, ALWAYS add cooking oil/butter card UNLESS boiled/steamed.
  - Omelette/bhurji: add 1 tsp butter (~45 kcal). Paratha: add 1 tsp ghee (~45 kcal).
  - NEVER skip fat for: omelette, paratha, poha, upma, sabzi, bhurji, stir-fry, fried rice.

MESS REALITY RULES (mandatory when user mentions "mess" or "canteen"):
- Mess Dal: Watery. Only 2-3g protein per katori (150g). Not 6-8g.
- Mess Sabzi: Heavy oil - add 10-15g fat (~90-135 kcal) per katori.
- Mess Roti: 37g each, plain, no ghee unless user says so.
- Mess Rice: 1 plate = ~200g cooked (~260 kcal). Not basmati.
- Canteen Chicken: Bone-in, ~40g edible per piece + ~45 kcal cooking fat per piece.
- Boiled Eggs: Zero oil. 1 large egg = 70 kcal, 6g protein.

PORTIONS & UNIT SCALING (Database & MCP results are per 100g — convert user units to grams first):
- 1 slice bread = 25g | 2 slices bread = 50g. NEVER calculate 1 slice as 100g! (e.g. if bread is ~260 kcal/100g, 2 slices is 50g = ~130 kcal).
- 1 katori = 150g | 1 roti / chapati = 37g | 1 plate cooked rice = 200g | 1 chicken piece (bone-in) = 40g edible
- 1 whole egg = 50g | 1 medium banana = 100g
- Spread on bread (butter, jam, chocolate spread) when unspecified = 15-20g total (scale per-100g MCP data to 15-20g).

LOGGING (INTENT A only):
1. get_food_macros per ingredient
2. log_meal_to_database (user_id from [CONTEXT], combined food_name, total cal, total protein)
3. Reply: $$MACRO cards first, then "Logged! Keep it up"
NEVER call log_meal_to_database for intents B or E.

FORMATTING RULES:
- Keep replies SHORT.
- $$MACRO cards are MANDATORY for every food item:
  $$MACRO: [Item Name] | [Calories] kcal | [Protein]g pro$$
- Add "(est.)" ONLY if DB returned NOT FOUND.
- Always add a $$MACRO card for manually added oil/ghee.

Example response:
$$MACRO: 150g Rice | 195 kcal | 4g pro$$
$$MACRO: 140g Chicken | 230 kcal | 42g pro$$
$$MACRO: Cooking Oil | 135 kcal | 0g pro$$
Total: 560 kcal | 46g protein. Logged!"""

NUTRITIONIST_SYSTEM_PROMPT = """You are FitDesi Desi Nutritionist - a warm, science-backed advisor for Indian students. You handle meal suggestions, fitness advice, supplement guidance, and personalized nutrition plans. You NEVER do meal logging (that is the Calculator job).

CURRENT USER CONTEXT (provided in messages): user_id, date, goal, targets, weight, diet. Always call get_user_profile FIRST before any personalized advice.

DIET RULE (check BEFORE every suggestion):
- diet=veg       -> ONLY vegetarian foods. No eggs, no meat, no fish. Ever.
- diet=eggetarian -> eggs OK. Absolutely NO meat or fish.
- diet=non-veg   -> all foods OK.
Suggesting food outside user diet type = critical error.

MEAL SUGGESTION FORMAT RULES:
Determine the user query type and format accordingly:

### MODE 1: IMMEDIATE / QUICK SNACK (<60 mins before workout/sports, or "quick bite" / "running late")
- NEVER use the 3-tier budget format for this.
- Recommend 1 direct, fast-digesting carb snack (e.g., 1-2 Bananas, 2 slices white bread with honey/jam, dates, or fruit).
- Scientific digestion rule: Clearly explain that heavy protein (chicken, paneer, eggs, whey) and fats/fiber (oats, peanut butter) take 2-4 hours to digest and divert blood to the gut, causing cramps and sluggishness. Fast-acting carbs supply immediate blood glucose and muscle glycogen without stomach load.
- Include $$MACRO cards for the suggested fast snack.

### MODE 2: SPECIFIC FOOD EVALUATION, FOOD SWAP, OR COMPARISON
(e.g., "Can I eat eggs before bed?", "Is Amul lassi good?", "Can I replace paneer with soya chunks?", "Whey vs Eggs")
- NEVER use the 3-tier budget format for this.
- Answer the specific question directly. If comparing or swapping, provide a direct side-by-side breakdown (protein per 100g, cost, digestibility, trade-offs).
- Include $$MACRO cards for the items discussed.

### MODE 3: PRECISION MACRO GAP FILLING
(e.g., "I have 250 kcal and 25g protein left today, what can I eat?")
- NEVER use the 3-tier budget format for this.
- Suggest 1-2 exact food options that fit the user remaining calorie and protein budget without overshooting.
- Include $$MACRO cards for the options.

### MODE 4: OPEN-ENDED MEAL PLANNING & BUDGET MEAL DISCOVERY
(e.g., "Suggest a high protein lunch", "Dinner options at hostel", "Pre-workout meal 2-3 hours before gym", "Cheap student meal plan")
- USE THE 3-TIER SYSTEM. Format EXACTLY:

**Tier 1 - Mess/Canteen (Rs 0-40)**
[one punchy sentence + $$MACRO cards]

**Tier 2 - Basic Grocery (Rs 40-100)**
[one punchy sentence + $$MACRO cards]

**Tier 3 - Slightly Premium (Rs 100+)**
[one punchy sentence + $$MACRO cards]

End with ONE trade-off line.

Tier content rules:
- Tier 1 ONLY: mess dal, roti, rice, sabzi, boiled eggs (mess), banana, canteen omelette.
- Tier 2 ONLY: oats, peanut butter, milk, curd, groundnuts, chana, bread, sprouts, soya chunks.
- Tier 3 ONLY: paneer, Greek yogurt, protein oats, chicken breast (supermarket), cottage cheese, whey (only if user says they have it).
- GLOBALLY BANNED (no tier): avocado, salmon, tuna steak, quinoa, acai.

--------------------------------------------------
GENERAL RULES FOR ALL MODES:
- Call get_food_macros for EVERY suggested item BEFORE replying.
- After your response, append:
  Keep in mind: (2-3 bullets, personalized from user profile - goal, protein target, timing, meal type)

SCIENCE AND SUPPLEMENT RULES:
1. ALWAYS call get_science_facts before answering any supplement or fitness myth question.
2. ALWAYS reference the user specific profile (Goal, Weight, Targets) when giving advice.
3. ANTI-HALLUCINATION RULE: Match RAG context to user goal. NEVER suggest a Bulking meal to a Cutting user.
4. Use NIN/JISSN standards (protein: 1.6-2.2g/kg bodyweight).
5. Cut -> warn on calorie excess. Bulk -> encourage more food.

FORMATTING RULES:
- $$MACRO cards are MANDATORY for every food item:
  $$MACRO: [Item Name] | [Calories] kcal | [Protein]g pro$$
- Add "(est.)" ONLY if DB returned NOT FOUND.
- Keep replies structured and warm, not clinical."""

GENERAL_SYSTEM_PROMPT = """You are FitDesi, a friendly Indian fitness AI for students.
For casual messages and greetings, reply warmly and briefly in Hinglish or English matching the user tone.
Keep it short and friendly. If they seem to have a fitness question, nudge them to ask properly."""


# ==========================================
# 8. AGENT NODE FUNCTIONS
# ==========================================

def manager_node(state: AgentState) -> dict:
    """Hub: classifies intent and sets routing. Uses cheap 8B model. No tools."""
    messages = state["messages"]

    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None
    )
    user_text = last_human.content if last_human else ""

    routing_messages = [
        SystemMessage(content=MANAGER_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]

    response = llm_8b.invoke(routing_messages)
    route = response.content.strip().lower().strip(".")

    if "calculator" in route:
        route = "calculator"
    elif "nutritionist" in route:
        route = "nutritionist"
    else:
        route = "general"

    print(f"[Manager] Routed to: {route}")
    return {"sender": route}


def calculator_node(state: AgentState) -> dict:
    """Spoke: handles macro math and meal logging."""
    context = state.get("context", "")
    dynamic_prompt = CALCULATOR_SYSTEM_PROMPT
    if context:
        dynamic_prompt += f"\n\n[CURRENT USER CONTEXT]\n{context}"

    full_messages = [SystemMessage(content=dynamic_prompt)] + list(state["messages"])
    response = llm_70b.bind_tools(calculator_tools).invoke(full_messages)

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        print(f"[Calculator] Token Usage: {response.usage_metadata}")

    return {"messages": [response], "sender": "calculator"}


def nutritionist_node(state: AgentState) -> dict:
    """Spoke: handles meal suggestions, science, and personalized advice."""
    context = state.get("context", "")
    dynamic_prompt = NUTRITIONIST_SYSTEM_PROMPT
    if context:
        dynamic_prompt += f"\n\n[CURRENT USER CONTEXT]\n{context}"

    full_messages = [SystemMessage(content=dynamic_prompt)] + list(state["messages"])
    response = llm_70b.bind_tools(nutritionist_tools).invoke(full_messages)

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        print(f"[Nutritionist] Token Usage: {response.usage_metadata}")

    return {"messages": [response], "sender": "nutritionist"}


def general_node(state: AgentState) -> dict:
    """Spoke: handles greetings and off-topic chat using cheap 8B model."""
    messages = [SystemMessage(content=GENERAL_SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_8b.invoke(messages)
    return {"messages": [response], "sender": "general"}


# ==========================================
# 9. ROUTING FUNCTIONS
# ==========================================

def route_from_manager(state: AgentState) -> Literal["calculator", "nutritionist", "general"]:
    """After Manager classifies, route to the right specialist."""
    return state.get("sender", "general")


def route_after_specialist(state: AgentState) -> Literal["tools", "__end__"]:
    """After Calculator or Nutritionist responds, check if they want to call a tool."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"


def route_after_tools(state: AgentState) -> Literal["calculator", "nutritionist"]:
    """After ToolNode executes, send results back to whoever called the tool."""
    return state.get("sender", "calculator")


# ==========================================
# 10. BUILD THE GRAPH (Hub-and-Spoke)
# ==========================================

memory = MemorySaver()

workflow = StateGraph(AgentState)

workflow.add_node("manager", manager_node)
workflow.add_node("calculator", calculator_node)
workflow.add_node("nutritionist", nutritionist_node)
workflow.add_node("general", general_node)
workflow.add_node("tools", tool_node)

# Hub: always hits Manager first
workflow.add_edge(START, "manager")

# Manager routes to the right spoke
workflow.add_conditional_edges(
    "manager",
    route_from_manager,
    {
        "calculator": "calculator",
        "nutritionist": "nutritionist",
        "general": "general",
    }
)

# Specialists: either call a tool or finish
workflow.add_conditional_edges(
    "calculator",
    route_after_specialist,
    {"tools": "tools", "__end__": END}
)
workflow.add_conditional_edges(
    "nutritionist",
    route_after_specialist,
    {"tools": "tools", "__end__": END}
)

# General chat always ends directly
workflow.add_edge("general", END)

# ToolNode: routes back to whoever called it (sender-aware)
workflow.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "calculator": "calculator",
        "nutritionist": "nutritionist",
    }
)

agent_executor = workflow.compile(checkpointer=memory)


# ==========================================
# 11. UTILITY FUNCTIONS (used by main.py)
# ==========================================

def generate_thread_title(user_message: str) -> str:
    """Generate a short, descriptive 3-5 word title from the user first message."""
    try:
        prompt = (
            f"Create a short chat title (3-5 words) for a fitness/nutrition conversation "
            f"starting with this message. Be specific. No quotes, no punctuation.\n"
            f"Examples: Eggs and Bread Macros, Creatine Advice, Bulking Diet Check\n"
            f"Message: \"{user_message.strip()[:200]}\"\n"
            f"Title:"
        )
        response = llm_8b.invoke([HumanMessage(content=prompt)])
        title = response.content.strip().strip("\"'").strip()
        return title[:60] if title else "New Conversation"
    except Exception as e:
        print(f"[Title Gen] Failed: {e}")
        return "New Conversation"


def present_response(raw_response: str) -> str:
    """Pure Python cleanup. Strips whitespace and collapses excessive blank lines."""
    cleaned = raw_response.strip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned


# ==========================================
# 12. TERMINAL TEST MODE
# ==========================================
if __name__ == "__main__":
    print("FitDesi Multi-Agent System is Online! (Type quit to exit)")
    print("  Manager (8B) -> Calculator / Nutritionist / General (70B)")
    print("-" * 60)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}
    test_context = "uid=test_user_01, date=2026-10-31, diet=eggetarian, wt=70kg, goal=cut, cal=1800, pro=140g"

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break

        print("\nThinking...\n")

        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "context": test_context,
            "sender": "",
        }

        chunk = None
        for chunk in agent_executor.stream(inputs, config=config, stream_mode="values"):
            last_msg = chunk["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"[TOOL] -> {tc['name']}({tc['args']})")

        if chunk:
            final = chunk["messages"][-1].content
            sender = chunk.get("sender", "?")
            print(f"\n[{sender.upper()}]: {present_response(final)}")
