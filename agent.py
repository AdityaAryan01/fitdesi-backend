import os
import re
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from database import SessionLocal
from models import FoodItem
from sqlalchemy import and_
from pydantic import BaseModel, Field

from datetime import date, timedelta, datetime, timezone
from zoneinfo import ZoneInfo
import models
import json

from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vector_store = Chroma(
    persist_directory="./data/chroma_db",
    embedding_function=embeddings
)

# ==========================================
# 1. TOOLS
# ==========================================

class FoodMacroInput(BaseModel):
    food_name: str = Field(
        ...,
        description="A SINGLE food item to search for (e.g., 'egg' OR 'bread'). NEVER pass combined meals."
    )

class ScienceQueryInput(BaseModel):
    query: str = Field(..., description="The scientific question, fitness myth, or supplement to verify.")

class MealHistoryInput(BaseModel):
    user_id: str = Field(..., description="The ID of the user")
    days: int = Field(default=1, description="How many days back to check (1=today, 7=last week)")


@tool(args_schema=FoodMacroInput)
def get_food_macros(food_name: str) -> str:
    """Search the database for exact food macros. Call ONCE per food item — never pass a combined meal string."""
    db = SessionLocal()
    try:
        words = food_name.lower().split()
        conditions = [FoodItem.item_name.ilike(f"%{word}%") for word in words]
        foods = db.query(FoodItem).filter(and_(*conditions)).limit(5).all()

        if not foods and len(words) > 1:
            for word in reversed(words):
                fallback = db.query(FoodItem).filter(
                    FoodItem.item_name.ilike(f"%{word}%")
                ).limit(3).all()
                if fallback:
                    foods = fallback
                    break

        if foods:
            results = [
                f"{f.item_name}: {f.calories} kcal, {f.protein}g protein, {f.carbs}g carbs, {f.fat}g fat per {f.serving_size}"
                for f in foods
            ]
            return "Found in database:\n" + "\n".join(results)

        return f"'{food_name}' NOT FOUND. Estimate using internal knowledge."
    except Exception:
        return "Database error. Please try again."
    finally:
        db.close()


@tool(args_schema=ScienceQueryInput)
def get_science_facts(query: str) -> str:
    """Verify scientific fitness facts, supplements, myths, or ICMR guidelines."""
    results = vector_store.similarity_search(query, k=3)
    if results:
        context = "\n\n".join([doc.page_content for doc in results])
        return f"Scientific context:\n{context}"
    return "No scientific context found."


class MealLogInput(BaseModel):
    user_id: str = Field(..., description="The user ID from CONTEXT")
    food_name: str = Field(..., description="Name of the food eaten")
    calories: str = Field(..., description="Total calories as a string e.g. '350'")
    protein: str = Field(..., description="Total protein as a string e.g. '12'")


@tool(args_schema=MealLogInput)
def log_meal_to_database(user_id: str, food_name: str, calories: str, protein: str) -> str:
    """Save a meal to the user's daily log. ONLY call when the user explicitly reports consuming food."""
    try:
        cal_val = float(calories)
        pro_val = float(protein)
    except ValueError:
        return "ERROR: Invalid calorie/protein values."

    if cal_val <= 0:
        return "ERROR: Calorie value cannot be zero or negative. Every real food has calories. Re-check get_food_macros or use a reasonable estimate (e.g. tea with milk ~35 kcal, plain tea ~2 kcal) and call log_meal_to_database again with the correct value."
    if cal_val > 3000:
        return "SECURITY BLOCK: Calorie value too high (>3000). Ask user to verify."
    if not (0 <= pro_val <= 300):
        return "ERROR: Invalid protein amount."

    clean_name = food_name.replace(";", "").replace("--", "").strip()

    # Always store the date in IST (UTC+5:30) — the user's actual local date
    today_ist = datetime.now(ZoneInfo('Asia/Kolkata')).date()

    db = SessionLocal()
    try:
        db.add(models.DailyLog(
            user_id=user_id,
            date=today_ist,
            food_name=clean_name,
            calories=round(cal_val, 1),
            protein=round(pro_val, 1)
        ))
        db.commit()
        return f"LOGGED: {clean_name} — {cal_val} kcal, {pro_val}g protein for user {user_id} on {today_ist}."
    except Exception as e:
        db.rollback()
        return f"DATABASE ERROR: {str(e)}"
    finally:
        db.close()


@tool(args_schema=MealHistoryInput)
def get_user_meal_history(user_id: str, days: int = 1) -> str:
    """Retrieve what the user ate in the past N days."""
    today_ist = datetime.now(ZoneInfo('Asia/Kolkata')).date()
    db = SessionLocal()
    try:
        start_date = today_ist - timedelta(days=days - 1)
        logs = db.query(models.DailyLog).filter(
            models.DailyLog.user_id == user_id,
            models.DailyLog.date >= start_date
        ).order_by(models.DailyLog.date.desc()).all()

        if not logs:
            return f"No meal logs found for user {user_id} in the last {days} day(s)."

        history = f"Meal Logs for {user_id}:\n"
        for log in logs:
            history += f"- [id={log.id}] {log.date}: {log.food_name} ({log.calories} kcal, {log.protein}g protein)\n"
        return history
    finally:
        db.close()


class UpdateMealLogInput(BaseModel):
    user_id: str = Field(..., description="The user ID from CONTEXT")
    food_name: str = Field(..., description="The food name to find and update (matches today's log)")
    new_calories: str = Field(..., description="Corrected calorie value as a string e.g. '35'")
    new_protein: str = Field(..., description="Corrected protein value as a string e.g. '1'")


@tool(args_schema=UpdateMealLogInput)
def update_meal_log(user_id: str, food_name: str, new_calories: str, new_protein: str) -> str:
    """Update an incorrectly logged meal entry for today. Use when the user says a logged value was wrong."""
    try:
        cal_val = float(new_calories)
        pro_val = float(new_protein)
    except ValueError:
        return "ERROR: Invalid calorie/protein values."

    if cal_val <= 0:
        return "ERROR: Corrected calorie value must be greater than 0."

    today_ist = datetime.now(ZoneInfo('Asia/Kolkata')).date()
    db = SessionLocal()
    try:
        # Find the most recent matching log entry for today
        log = (
            db.query(models.DailyLog)
            .filter(
                models.DailyLog.user_id == user_id,
                models.DailyLog.date == today_ist,
                models.DailyLog.food_name.ilike(f"%{food_name.strip()}%")
            )
            .order_by(models.DailyLog.id.desc())
            .first()
        )
        if not log:
            return f"No log entry found for '{food_name}' today. Use get_user_meal_history to see what was logged."

        old_cal = log.calories
        old_pro = log.protein
        log.calories = round(cal_val, 1)
        log.protein = round(pro_val, 1)
        db.commit()
        return f"UPDATED: '{log.food_name}' — changed from {old_cal} kcal/{old_pro}g protein to {cal_val} kcal/{pro_val}g protein."
    except Exception as e:
        db.rollback()
        return f"DATABASE ERROR: {str(e)}"
    finally:
        db.close()


# ==========================================
# 2. LLM SETUP
# ==========================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# Fast model for intent + language detection
llm_fast = ChatGroq(
    model="llama-3.3-70b-versatile",   # Use same model — 8b is too weak for instructions
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

tools = [get_food_macros, get_science_facts, log_meal_to_database, update_meal_log, get_user_meal_history]
memory = MemorySaver()


def generate_thread_title(user_message: str) -> str:
    """Generate a short, descriptive 3-6 word title from the user's first message."""
    try:
        prompt = (
            f"Create a short chat title (3-6 words) for a fitness/nutrition conversation "
            f"starting with this message. Be specific and descriptive. "
            f"No quotes, no punctuation at the end, no filler words like 'question about'.\n"
            f"Examples: 'Dal protein content query', 'Pre-workout meal plan', 'Evening snack macros', 'Creatine supplement advice'\n"
            f"Message: \"{user_message.strip()[:200]}\"\n"
            f"Title:"
        )
        response = llm_fast.invoke([HumanMessage(content=prompt)])
        title = response.content.strip().strip('"\'').strip()
        # Truncate to max 60 chars for safety
        return title[:60] if title else "New Conversation"
    except Exception as e:
        print(f"[Title Gen] Failed: {e}")
        return "New Conversation"

BASE_SYSTEM_PROMPT = """You are FitDesi, a fitness and nutrition assistant for Indian hostel diets.

USER PROFILE:
{user_profile_section}

DIET RESTRICTION (STRICTLY FOLLOW - NO EXCEPTIONS):
{diet_rules_section}

RULES:
- Use get_food_macros for food calories/macros. Never guess without trying the tool first.
- ZERO CALORIES IS NEVER ACCEPTABLE for any real food. Plain tea = ~2 kcal, tea with milk+sugar = ~35 kcal. If get_food_macros returns 0 or nothing, use a sensible estimate greater than 0.
- Use get_science_facts ONLY for supplement/fitness science/myth questions. Do NOT use it for calorie targets or general nutrition advice - answer those directly.
- Use generic homemade estimates unless user mentions a brand.
- Indian units: 1 katori=150g, 1 roti=37g, 1 plate rice=200g.
- Messages include [CONTEXT: user_id=X, date=YYYY-MM-DD]. Always extract user_id from there.
- Call log_meal_to_database ONLY when user says they ate something right now. Never log for questions.
- Logging flow: get_food_macros per item, then log_meal_to_database, then reply.
- If log_meal_to_database returns an error about zero/invalid calories, call get_food_macros again and use a non-zero estimate.
- For past meal queries, call get_user_meal_history with user_id from CONTEXT.
- If the user says a logged value was wrong or needs correction, call update_meal_log with the correct values. Do NOT log a new entry.
"""

DIET_RULES = {
    "veg": (
        "The user is VEGETARIAN. This is an absolute restriction:\n"
        "- NEVER suggest, recommend, or mention: chicken, mutton, fish, egg, seafood, beef, pork, or any meat/poultry/seafood.\n"
        "- ALL suggestions MUST be 100% vegetarian: paneer, dal, rajma, chana, tofu, milk, curd, whey protein, soya, nuts, seeds, vegetables, fruits, grains.\n"
        "- If asked for high-protein foods, ONLY suggest vegetarian sources.\n"
        "- If the user mentions eating a non-veg food, log it without judgment but do not recommend more non-veg."
    ),
    "eggetarian": (
        "The user is EGGETARIAN (vegetarian + eggs).\n"
        "- NEVER suggest: chicken, mutton, fish, seafood, beef, pork, or any meat/poultry.\n"
        "- Eggs ARE allowed and can be suggested.\n"
        "- ALL suggestions must be vegetarian or egg-based."
    ),
    "non-veg": (
        "The user eats non-vegetarian food. All food types are acceptable."
    ),
}


def build_system_prompt(user_profile: dict | None) -> str:
    """Builds a personalized system prompt with the user's diet, goal, and macro targets injected."""
    if not user_profile:
        # Fallback: no profile info — use safe defaults
        diet_rules = DIET_RULES["veg"]  # Default to veg as the safer option
        profile_section = "User profile not loaded. Assume vegetarian by default."
    else:
        diet_type = (user_profile.get("diet_type") or "veg").lower().strip()
        goal = user_profile.get("goal", "maintenance")
        target_cal = user_profile.get("target_calories", 2000)
        target_pro = user_profile.get("target_protein", 120)
        name = user_profile.get("name", "the user")
        weight = user_profile.get("weight_kg", "?")
        diet_rules = DIET_RULES.get(diet_type, DIET_RULES["veg"])
        profile_section = (
            f"- Name: {name}\n"
            f"- Goal: {goal}\n"
            f"- Diet type: {diet_type}\n"
            f"- Daily calorie target: {target_cal} kcal\n"
            f"- Daily protein target: {target_pro}g\n"
            f"- Weight: {weight} kg\n"
            f"Always align your recommendations with this goal and these targets."
        )

    return BASE_SYSTEM_PROMPT.format(
        user_profile_section=profile_section,
        diet_rules_section=diet_rules,
    )


# Default executor (no profile) — overridden per-request in main.py
_default_prompt = build_system_prompt(None)
agent_executor = create_agent(
    llm,
    tools,
    system_prompt=_default_prompt,
    checkpointer=memory
)


# ==========================================
# 3. LANGUAGE DETECTION (Rule-Based — Fast & Reliable)
# ==========================================

def detect_language(text: str) -> str:
    """
    Reliably detect if the user's message is English or Hindi.
    Uses Unicode script detection — does NOT rely on an LLM.

    Returns: 'english' or 'hindi'
    """
    # Count Devanagari characters (Hindi script)
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))

    if devanagari_chars > 0:
        return 'hindi_devanagari'  # Actual Hindi script

    # For Romanized Hindi: check if the sentence STRUCTURE is Hindi
    # We do this by looking for strong Hindi-only function words
    HINDI_MARKERS = [
        r'\bkya\b', r'\bhai\b', r'\btha\b', r'\bkhaaya\b', r'\bkhayi\b',
        r'\bkiya\b', r'\bkaro\b', r'\bkarte\b', r'\bmera\b', r'\bmeri\b',
        r'\bnahi\b', r'\bnahi\b', r'\bhun\b', r'\bhoon\b', r'\baur\b',
        r'\bkuch\b', r'\bkoi\b', r'\bwoh\b', r'\bwoh\b', r'\bapna\b',
        r'\bthoda\b', r'\bbata\b', r'\bbhej\b', r'\bde\b', r'\bdo\b',
        r'\bpuchh\b', r'\blikha\b', r'\bpadh\b', r'\bsun\b'
    ]
    text_lower = text.lower()
    hindi_marker_count = sum(1 for pattern in HINDI_MARKERS if re.search(pattern, text_lower))

    # If 2+ Hindi-specific words found, treat as Hindi
    if hindi_marker_count >= 2:
        return 'hindi_roman'

    return 'english'


# ==========================================
# 4. INTENT DETECTION
# ==========================================

INTENT_PROMPT = """You are an intent classifier for a fitness tracking app.

Analyze this message and determine if the user is EXPLICITLY reporting they ate, drank, or consumed something RIGHT NOW.

Respond ONLY with valid JSON, nothing else:
{{"is_meal_log": true}} OR {{"is_meal_log": false}}

TRUE only if: user clearly states consumption right now (e.g., "I ate 2 eggs", "had chai", "just finished lunch", "khaaya 2 roti", "2 piece mushroom 3 potato slices")
FALSE for: questions about food, hypotheticals, general queries, past meal checks

Message: "{message}"

JSON:"""


# Quick keyword-based pre-check (runs before any LLM call)
# If the message contains clear food consumption signals, we mark it as True immediately.
# This runs in microseconds and is immune to API failures.
FOOD_CONSUMPTION_KEYWORDS = [
    r'\bate\b', r'\beat\b', r'\bhad\b', r'\bdrank\b', r'\bdrink\b',
    r'\bfinished\b', r'\bjust ate\b', r'\bjust had\b', r'\bjust drank\b',
    r'\bkhaaya\b', r'\bkhayi\b', r'\bkha liya\b', r'\bpiya\b',
    r'\bbreakfast\b', r'\blunch\b', r'\bdinner\b', r'\bsnack\b',
    # Unit patterns that strongly imply food consumption
    r'\d+\s*(piece|pieces|cup|cups|bowl|plate|katori|roti|egg|eggs|glass|scoop|grams?|g)\b',
    # Common Indian food names that imply consumption when mentioned with quantity
]

def _keyword_is_food(text: str) -> bool:
    """Returns True if text contains clear food consumption signals."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in FOOD_CONSUMPTION_KEYWORDS)


def detect_meal_intent(user_message: str) -> bool:
    """
    Returns True if user is logging a meal.
    
    Strategy:
    1. First, do a fast keyword pre-check. If clearly food → True immediately.
    2. Otherwise, call the LLM classifier.
    3. CRITICAL: Default to True on any error — the core agent's own prompt
       already guards against logging non-food messages. It is far better to
       allow the agent to decide than to silently block correct logging.
    """
    # Step 1: Fast keyword pre-check
    if _keyword_is_food(user_message):
        print(f"[Intent Detection] Keyword match — is_meal_log=True (skipped LLM)")
        return True

    # Step 2: LLM classifier
    try:
        prompt = INTENT_PROMPT.format(message=user_message.replace('"', "'"))
        response = llm_fast.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        result = json.loads(raw)
        intent = bool(result.get("is_meal_log", True))  # Default True in JSON parse too
        print(f"[Intent Detection] LLM classified — is_meal_log={intent}")
        return intent
    except Exception as e:
        # SAFE DEFAULT: True (allow logging). The core agent will NOT log if the message
        # is clearly not about food — it has its own guard in the system prompt.
        print(f"[Intent Detection] Error: {e} — defaulting to True (allow agent to decide)")
        return True


# ==========================================
# 5. FOOD BREAKDOWN AGENT
# ==========================================

BREAKDOWN_PROMPT = """You are a nutritional analysis agent for a fitness app.

User diet restriction: {diet_type}
{diet_rule}

The user mentioned eating these foods: "{food_items_text}"

Your job:
1. Identify each individual food item and its quantity from the input.
2. For each item, provide: item name, quantity, estimated calories, estimated protein.
3. At the bottom, provide a TOTAL row.
4. IMPORTANT: Only analyze what the user actually said they ate. Do NOT suggest alternative foods here.

Use Indian standard estimates. Be precise — give single numbers, not ranges.

Respond ONLY with a JSON array and nothing else:
[
  {{"item": "food name", "quantity": "2 pieces", "calories": 120, "protein": 4}},
  {{"item": "another food", "quantity": "1 scoop", "calories": 200, "protein": 5}},
  {{"total": true, "calories": 320, "protein": 9}}
]

Foods to analyze: "{food_items_text}"

JSON array:"""


def get_food_breakdown(food_items_text: str, diet_type: str = "veg") -> list[dict] | None:
    """
    Breaks down a meal description into per-item nutritional data.
    Returns a list of dicts with item, quantity, calories, protein.
    Returns None if it fails.
    """
    try:
        diet_rule = DIET_RULES.get(diet_type.lower().strip(), DIET_RULES["veg"])
        prompt = BREAKDOWN_PROMPT.format(
            food_items_text=food_items_text.replace('"', "'"),
            diet_type=diet_type,
            diet_rule=diet_rule,
        )
        response = llm_fast.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return None
    except Exception as e:
        print(f"[Food Breakdown Agent] Error: {e}")
        return None


# ==========================================
# 6. PRESENTER AGENT
# ==========================================

PRESENTER_PROMPT_ENGLISH = """You are a response formatter for a fitness assistant app.

The user wrote in ENGLISH. You MUST respond ENTIRELY in English.
- Do NOT use any Hindi, Devanagari script, or Hindi words whatsoever.
- Not even words like "khaana", "bhai", "yaar", "roti" unless quoting the user.
- Technical terms are fine: calories, protein, kcal, grams.

USER DIET RESTRICTION (CRITICAL — MUST ENFORCE):
{diet_rule}
If the raw response contains any food suggestions that violate the above diet, SILENTLY REMOVE them and replace with appropriate alternatives before formatting. Never show forbidden foods to the user.

Rules:
- Professional, clear, helpful tone. No slang.
- Use markdown: **bold** for numbers, bullet points for lists.
- If meal was logged: brief 1-2 sentence confirmation only.
- Do NOT change any numbers from the raw response.
- Only reformat, clean tone, and enforce diet rules.

{breakdown_section}

Raw AI response to reformat:
"{raw_response}"

Final English response:"""

PRESENTER_PROMPT_HINDI_ROMAN = """You are a response formatter for a fitness assistant app.

The user wrote in Romanized Hindi (Hindi words using English letters). You MUST respond in Romanized Hindi ONLY.
- Use Hindi words written in English letters (e.g., "Aapka khaana", "Kal aapne khaaya").
- Do NOT use Devanagari script at all.
- Only English allowed: technical terms like calories, protein, kcal, grams.
- No Hinglish sentence mixing — full Hindi sentences in Roman script.
- Professional tone. No "bhai", "yaar" etc.

USER DIET RESTRICTION (CRITICAL — MUST ENFORCE):
{diet_rule}
If the raw response contains any food suggestions that violate the above diet, SILENTLY REMOVE them and replace with appropriate alternatives before formatting.

{breakdown_section}

Raw AI response to reformat:
"{raw_response}"

Final Romanized Hindi response:"""

PRESENTER_PROMPT_HINDI_DEV = """You are a response formatter for a fitness assistant app.

The user wrote in Hindi (Devanagari). You MUST respond in proper Hindi using Devanagari script.
- Professional, helpful tone. No slang.
- Only English allowed: calories, protein, kcal, grams.
- Use markdown: **bold** for numbers, bullet points.

USER DIET RESTRICTION (CRITICAL — MUST ENFORCE):
{diet_rule}
If the raw response contains any food suggestions that violate the above diet, SILENTLY REMOVE them and replace with appropriate alternatives before formatting.

{breakdown_section}

Raw AI response to reformat:
"{raw_response}"

Final Hindi (Devanagari) response:"""


def _format_breakdown_section(breakdown: list[dict] | None) -> str:
    """Formats the breakdown data into a readable markdown section for the presenter."""
    if not breakdown:
        return ""

    lines = ["**Nutritional Breakdown:**", ""]
    total_row = None

    for item in breakdown:
        if item.get("total"):
            total_row = item
            continue
        name = item.get("item", "Unknown")
        qty = item.get("quantity", "")
        cal = item.get("calories", "?")
        pro = item.get("protein", "?")
        lines.append(f"- **{name}** ({qty}): **{cal} kcal**, {pro}g protein")

    if total_row:
        lines.append("")
        lines.append(f"**Total: {total_row.get('calories', '?')} kcal | {total_row.get('protein', '?')}g protein**")

    return "\n".join(lines)


def present_response(user_message: str, raw_response: str, breakdown: list[dict] | None = None, user_profile: dict | None = None) -> str:
    """
    Final quality-control pass:
    1. Detects language via rule-based method (reliable).
    2. Picks the right language-specific prompt template.
    3. Injects the food breakdown table if available.
    4. Injects diet restrictions so forbidden foods are scrubbed from output.
    5. Runs the presenter LLM call.
    """
    lang = detect_language(user_message)
    print(f"[Presenter] Detected language: {lang}")

    # Resolve diet rule for this user
    diet_type = (user_profile or {}).get("diet_type", "veg") or "veg"
    diet_rule = DIET_RULES.get(diet_type.lower().strip(), DIET_RULES["veg"])

    breakdown_section = _format_breakdown_section(breakdown)
    if breakdown_section:
        breakdown_section = f"\nInclude this breakdown in your response:\n{breakdown_section}\n"

    try:
        if lang == 'english':
            prompt_template = PRESENTER_PROMPT_ENGLISH
        elif lang == 'hindi_roman':
            prompt_template = PRESENTER_PROMPT_HINDI_ROMAN
        else:  # hindi_devanagari
            prompt_template = PRESENTER_PROMPT_HINDI_DEV

        prompt = prompt_template.format(
            breakdown_section=breakdown_section,
            raw_response=raw_response.replace('"', "'"),
            diet_rule=diet_rule,
        )

        response = llm_fast.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"[Presenter Agent] Error: {e} — returning raw")
        return raw_response


# ==========================================
# 7. CLI TEST
# ==========================================
if __name__ == "__main__":
    print("💪 FitDesi Agent Online (type 'quit' to exit)")
    print("-" * 50)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break

        lang = detect_language(user_input)
        print(f"[Language] {lang}")

        is_logging = detect_meal_intent(user_input)
        print(f"[Intent] is_meal_log = {is_logging}")

        breakdown = None
        if is_logging:
            print("[Breakdown] Analyzing food items...")
            breakdown = get_food_breakdown(user_input)
            print(f"[Breakdown] {breakdown}")

        inputs = {"messages": [HumanMessage(content=f"[CONTEXT: user_id=test, date={date.today()}]\n{user_input}")]}
        result = agent_executor.invoke(inputs, config=config)
        raw = result["messages"][-1].content
        print(f"\n[RAW]: {raw[:200]}...")

        final = present_response(user_input, raw, breakdown)
        print(f"\n💪 FitDesi: {final}")