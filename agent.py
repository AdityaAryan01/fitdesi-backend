import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, trim_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Sequence
import operator
import typing
from database import SessionLocal
from models import FoodItem
from sqlalchemy import and_
from pydantic import BaseModel, Field, field_validator

from datetime import date, timedelta
import models

from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vector_store = Chroma(
    persist_directory="./data/chroma_db",
    embedding_function=embeddings
)

# ==========================================
# 1. DEFINE THE TOOLS (The Agent's "Hands")
# ==========================================

class FoodMacroInput(BaseModel):
    food_name: str = Field(
        ..., 
        description="A SINGLE food item to search for (e.g., 'egg' OR 'bread' OR 'amul cheese'). NEVER pass combined meals."
    )

class ScienceQueryInput(BaseModel):
    query: str = Field(
        ..., 
        description="The scientific question, fitness myth, or supplement to verify."
    )

class MealHistoryInput(BaseModel):
    user_id: str = Field(..., description="The ID of the user")
    days: int = Field(default=1, description="How many days back to check (1=today, 7=last week)")


@tool(args_schema=FoodMacroInput)
def get_food_macros(food_name: str) -> str:
    """Use this tool to search the database for exact food macros.
    
    CRITICAL SEARCH RULES:
    1. ATOMIC SEARCHES ONLY: You MUST pass ONLY ONE single food item at a time (e.g., 'egg' OR 'bread' OR 'amul cheese').
    2. NEVER pass combined meals or long phrases (e.g., DO NOT pass 'cheese omelette bread 4 eggs'). 
    3. MULTIPLE CALLS: If the user meal has 3 ingredients, you MUST call this tool 3 separate times.
    4. If the database returns 'NOT FOUND', DO NOT search for that exact item again. Estimate it yourself.
    """
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
            results = []
            for f in foods:
                results.append(f"{f.item_name}: {f.calories} kcal, {f.protein}g protein, {f.carbs}g carbs, {f.fat}g fat per {f.serving_size}")
            return "Found these matches in the database:\n" + "\n".join(results)
        
        return f"'{food_name}' NOT FOUND. STOP searching for this. Estimate macros based on your internal knowledge immediately."
    except Exception as e:
        return f"Database error occurred. Please try again."
    finally:
        db.close()



@tool(args_schema=ScienceQueryInput)
def get_science_facts(query: str) -> str:
    """Use this tool to verify scientific fitness facts, supplements, myths, or ICMR guidelines."""
    results = vector_store.similarity_search(query, k=2)  # already changed to k=2
    
    if results:
        context = "\n\n".join([doc.page_content[:400] for doc in results])  # ← only change
        return f"Scientific Context retrieved:\n{context}"
    return "No scientific context found in the research papers for this query."

class MealLogInput(BaseModel):
    user_id: str = Field(..., description="The ID of the user")
    food_name: str = Field(..., description="Name of the food eaten")
    calories: str = Field(..., description="Total calories. Provide as a string, e.g. '200'")
    protein: str = Field(..., description="Total protein. Provide as a string, e.g. '12'")


@tool(args_schema=MealLogInput)
def log_meal_to_database(user_id: str, food_name: str, calories: str, protein: str) -> str:
    """Use this tool to SAVE a meal to the user's daily log AFTER you have calculated the macros.
    You MUST call this tool whenever the user explicitly ate something (past tense confirmed consumption).
    NEVER call this for hypothetical, curiosity, or future meals.
    """
    try:
        cal_val = float(calories)
        pro_val = float(protein)
    except ValueError:
        return "ERROR: Calories and protein could not be calculated. Try again."

    if cal_val < 0 or cal_val > 3000:
        return "SECURITY BLOCK: Calories exceed realistic single meal limit (3000 kcal). Ask user to verify."
    if pro_val < 0 or pro_val > 300:
        return "ERROR: Invalid protein amount."

    clean_food_name = food_name.replace(";", "").replace("--", "").strip()

    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            return f"DATABASE ERROR: User {user_id} not found."

        # Auto-start day if not active
        if not user.active_tracking_date:
            from datetime import datetime
            user.active_tracking_date = date.today()
            user.active_tracking_start = datetime.utcnow()
            db.commit()

        new_log = models.DailyLog(
            user_id=user_id,
            date=user.active_tracking_date,
            food_name=clean_food_name,
            calories=round(cal_val, 1),
            protein=round(pro_val, 1)
        )
        db.add(new_log)
        db.commit()
        return f"SUCCESS: Securely logged {clean_food_name} ({cal_val} kcal) for user {user_id}."
    
    except Exception as e:
        db.rollback()
        return f"DATABASE ERROR: Failed to log meal. Detail: {str(e)}"
    finally:
        db.close()


@tool(args_schema=MealHistoryInput)
def get_user_meal_history(user_id: str, days: int = 1) -> str:
    """Use this tool to check what the user ate in the past. 
    'days' is how many days back to check (e.g., 1 for today, 7 for last week).
    """
    db = SessionLocal()
    try:
        start_date = date.today() - timedelta(days=days-1)
        
        logs = db.query(models.DailyLog).filter(
            models.DailyLog.user_id == user_id,
            models.DailyLog.date >= start_date
        ).order_by(models.DailyLog.date.desc()).all()

        if not logs:
            return f"Bhai, {user_id} ke liye pichle {days} dinon mein koi logs nahi mile."

        history_text = f"Recent Meal Logs for {user_id}:\n"
        for log in logs:
            history_text += f"- {log.date}: {log.food_name} ({log.calories} kcal, {log.protein}g protein)\n"
        
        return history_text
    finally:
        db.close()


# ==========================================
# 2. SETUP THE BRAIN
# ==========================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# --- NEW: Fast Brain for formatting and titling (Saves massive tokens) ---
llm_fast = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

def generate_thread_title(user_message: str) -> str:
    """Generate a short, descriptive 3-5 word title from the user's first message."""
    try:
        prompt = (
            f"Create a short chat title (3-5 words) for a fitness/nutrition conversation "
            f"starting with this message. Be specific. No quotes, no punctuation.\n"
            f"Examples: 'Eggs and Bread Macros', 'Creatine Advice', 'Bulking Diet Check'\n"
            f"Message: \"{user_message.strip()[:200]}\"\n"
            f"Title:"
        )
        response = llm_fast.invoke([HumanMessage(content=prompt)])
        

        title = response.content.strip().strip('"\'').strip()
        return title[:60] if title else "New Conversation"
    except Exception as e:
        print(f"[Title Gen] Failed: {e}")
        return "New Conversation"

def present_response(raw_response: str) -> str:
    """Pure Python cleanup — zero LLM tokens. Strips leading/trailing whitespace
    and collapses 3+ consecutive blank lines to 2. The 70B agent already
    outputs good markdown so we don't need a second LLM call."""
    import re
    cleaned = raw_response.strip()
    # Collapse excessive blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned

tools = [get_food_macros, get_science_facts, log_meal_to_database, get_user_meal_history]
memory = MemorySaver()

system_prompt = """You are FitDesi, an Indian gym bro AI for hostel students. Reply in English or Hinglish matching the user.

[CONTEXT] tag at message start contains: user_id, date, goal, targets, weight, diet. Extract user_id from it for all tool calls.

INTENT RULES (classify first, act second):
A) ATE IT (past tense: "khaaya", "had", "ate", "kha liya", "finished") → get_food_macros per ingredient → log_meal_to_database → reply with macros + confirm
B) MACRO QUERY ("calories in", "protein of", "kitna protein", "is X healthy") → get_food_macros → reply only. NEVER log.
C) FUTURE/HYPOTHETICAL/SUGGESTION ("planning to", "should I eat", "suggest a meal") → ALL suggestions MUST be Indian/Desi. ALWAYS call get_science_facts FIRST to find Indian meal options, then devise meal → advise only. NEVER log.
D) SCIENCE/FITNESS Q → get_science_facts → answer
E) PAST MEALS ("aaj kya khaaya", "show logs") → get_user_meal_history → summarize
AMBIGUOUS (no verb, e.g. "2 eggs") → ask "Khaaya ya sirf check karna tha?"

SEARCH RULES:
- One ingredient per get_food_macros call. Never combine.
- If no result: translate Hindi↔English and retry once. Still nothing → ask user for ingredients.
- Prefer homemade/mess food over branded unless user specifies brand.
- Never assume zero calories for cooked dishes.

PORTIONS (database is per 100g):
1 katori=150g | 1 roti=37g | 1 plate rice=200g | 1 chicken piece (bone-in)=40g edible

LOGGING (INTENT A only):
1. Call `get_food_macros` tool for each ingredient (optional if you are absolutely sure of the macros).
2. YOU MUST ACTUALLY CALL AND EXECUTE THE `log_meal_to_database` TOOL. Do not just write that you logged it. You must generate the tool call JSON. Use the user_id from [CONTEXT], combined food_name, total cal, total protein.
3. Only after the `log_meal_to_database` tool has executed successfully, reply to the user with the macros and "Logged! Keep it up 💪"
Never log for B/C/D/E intents.

ADVICE: Use [CONTEXT] profile for personalization. Cut→warn on excess. Bulk→encourage more. Use NIN/JISSN standards (protein: 1.6-2.2g/kg).
"""


# ==========================================
# 2b. BUILD THE GRAPH (replaces create_agent)
# ==========================================

class State(typing.TypedDict):
    messages: Annotated[Sequence[AnyMessage], operator.add]

def call_model(state: State):
    # Trim to last 12 messages before sending to LLM
    trimmed = trim_messages(
        messages=state["messages"],
        max_tokens=12,
        strategy="last",
        token_counter=len,       # count by number of messages, not actual tokens
        include_system=True,
        allow_partial=False,
    )
    # Prepend system prompt + invoke
    full_messages = [SystemMessage(content=system_prompt)] + list(trimmed)
    response = llm.bind_tools(tools).invoke(full_messages)

    return {"messages": [response]}
workflow = StateGraph(State)
workflow.add_node("model", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "model")
workflow.add_conditional_edges("model", tools_condition)
workflow.add_edge("tools", "model")

agent_executor = workflow.compile(checkpointer=memory)

# ==========================================
# 3. TEST THE AGENT IN THE TERMINAL
# ==========================================
if __name__ == "__main__":
    print("💪 FitDesi Gym Bro Agent is Online! (Type 'quit' to exit)")
    print("-" * 50)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        print("\nGym Bro is thinking... 🤔 (and calling tools if needed)\n")

        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        for chunk in agent_executor.stream(inputs, config=config, stream_mode="values"):
            message = chunk["messages"][-1]
            
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"🔧 [TOOL] -> {tool_call['name']}")
                
        # Safe content extraction
        content = getattr(message, "content", None)
        if content is None:
            content = message[1] if isinstance(message, tuple) else str(message)

        print(f"\n💪 Gym Bro: {content}")