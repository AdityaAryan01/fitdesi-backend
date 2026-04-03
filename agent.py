import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent  
from database import SessionLocal
from models import FoodItem
from sqlalchemy import and_
from pydantic import BaseModel, Field, field_validator

from datetime import date, timedelta
import models

from langgraph.checkpoint.memory import MemorySaver # ← Import the MemorySaver checkpoint

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

# ✅ FIX 1: Added MealHistoryInput schema (was missing before)
class MealHistoryInput(BaseModel):
    user_id: str = Field(..., description="The ID of the user")
    days: int = Field(default=1, description="How many days back to check (1=today, 7=last week)")

# --- YOUR EXACT EXISTING TOOLS, NOW WITH args_schema ---

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
        # Smart search: Split words and search independently
        words = food_name.lower().split()
        conditions = [FoodItem.item_name.ilike(f"%{word}%") for word in words]
        
        foods = db.query(FoodItem).filter(and_(*conditions)).limit(5).all()

        if not foods and len(words) > 1:
            # Try each word separately, return first match found
            for word in reversed(words):  # reversed = last word first (usually the food)
                fallback = db.query(FoodItem).filter(
                    FoodItem.item_name.ilike(f"%{word}%")
                ).limit(3).all()
                
                if fallback:
                    foods = fallback
                    break  # Stop as soon as we find something useful

        if foods:
            results = []
            for f in foods:
                results.append(f"{f.item_name}: {f.calories} kcal, {f.protein}g protein, {f.carbs}g carbs, {f.fat}g fat per {f.serving_size}")
            return "Found these matches in the database:\n" + "\n".join(results)
        
        return f"'{food_name}' NOT FOUND. STOP searching for this. Estimate macros based on your internal knowledge immediately."
    except Exception as e:
        return f"Database error occurred. Please try again."  # ✅ graceful error handling without exposing details
    finally:
        db.close()


# ✅ FIX 2: Added args_schema=ScienceQueryInput (was missing before — root cause of the 400 error)
@tool(args_schema=ScienceQueryInput)
def get_science_facts(query: str) -> str:
    """Use this tool to verify scientific fitness facts, supplements, myths, or ICMR guidelines."""
    # We no longer need to initialize embeddings inside here because it's global now!
    results = vector_store.similarity_search(query, k=3) 
    
    if results:
        context = "\n\n".join([doc.page_content for doc in results])
        return f"Scientific Context retrieved:\n{context}"
    return "No scientific context found in the research papers for this query."


# --- THE SECURE TOOL (PYDANTIC SHIELD) ---
class MealLogInput(BaseModel):
    user_id: str = Field(..., description="The ID of the user")
    food_name: str = Field(..., description="Name of the food eaten")
    # Tell Groq to expect strings, so it never crashes on quotes again
    calories: str = Field(..., description="Total calories. Provide as a string, e.g. '200'")
    protein: str = Field(..., description="Total protein. Provide as a string, e.g. '12'")

# --- THE SECURE TOOL ---
@tool(args_schema=MealLogInput)
def log_meal_to_database(user_id: str, food_name: str, calories: str, protein: str) -> str:
    """Use this tool to SAVE a meal to the user's daily log AFTER you have calculated the macros.
    You MUST call this tool whenever the user explicitly eats something.
    """
    # 1. Safely convert the AI's strings into math numbers
    try:
        cal_val = float(calories)
        pro_val = float(protein)
    except ValueError:
        return "ERROR: Calories and protein could not be calculated. Try again."

    # 2. Python Business Logic Validation
    if cal_val < 0 or cal_val > 3000:
        return "SECURITY BLOCK: Calories exceed realistic single meal limit (3000 kcal). Ask user to verify."
    if pro_val < 0 or pro_val > 300:
        return "ERROR: Invalid protein amount."

    # 3. Sanitize Text
    clean_food_name = food_name.replace(";", "").replace("--", "").strip()

    db = SessionLocal()
    try:
        # 4. Secure Database Write
        new_log = models.DailyLog(
            user_id=user_id,
            date=date.today(),
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


# ✅ FIX 3: Added args_schema=MealHistoryInput (was missing before)
@tool(args_schema=MealHistoryInput)
def get_user_meal_history(user_id: str, days: int = 1) -> str:
    """Use this tool to check what the user ate in the past. 
    'days' is how many days back to check (e.g., 1 for today, 7 for last week).
    """
    db = SessionLocal()
    try:
        # Calculate the date range
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
# 2. SETUP THE BRAIN (Azure OpenAI)
# ==========================================

# Initialize the Chat Model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,  # ✅ FIX 4: Changed from 0.3 to 0 — higher temp causes hallucinated tool call formats
)

# Bind tools to the agent
tools = [get_food_macros, get_science_facts, log_meal_to_database, get_user_meal_history]

# --- NEW: Initialize Memory ---
memory = MemorySaver()

# The System Prompt (Persona)
system_prompt = """You are FitDesi, a strict but helpful Indian gym bro and health assistant. 
You are talking to an Indian college student living in a hostel/PG.
Respond in the same way as the user, English or Hinglish whatever the user uses. Be highly practical, budget-conscious, and motivational.

--- 1. CORE BEHAVIOR & THE HOSTEL RULE ---
1. TOOLS FIRST: Always use 'get_food_macros' for nutrition/calories and 'get_science_facts' for fitness science/myths. DO NOT guess macros.
2. THE HOSTEL DEFAULT RULE: If the user DOES NOT mention a brand (like KFC or Domino's), you MUST prioritize generic, homemade, or mess food from the database results. Even if not given in the database, use your brain to estimate dish macros according to mess standardsq. NEVER use fast-food brand data unless explicitly requested.
3. NO ZERO CALORIES: Never assume a cooked dish (like 'potato masala') has negligible/zero calories. If you can't find the exact dish, search for its raw ingredients (e.g., potato + oil) or ask the user how it was made.
4. RESPONSE STRUCTURE: When the user logs food, always present the macro breakdown FIRST. Only after giving the numbers, confirm that you have saved it. Keep the confirmation brief and direct. 

--- 2. SEARCH & FALLBACK RULES ---
1. BEST MATCH: If 'get_food_macros' returns multiple options, pick the one that best matches the user's description.
2. TRANSLATE & RETRY: If your first search returns 0 results (e.g., 'potato masala' fails), TRANSLATE the keyword to its Hindi/English counterpart (e.g., 'aloo') and CALL THE TOOL AGAIN before giving up.
3. ASK FOR DETAILS: If the database still has no results after retrying, DO NOT hallucinate numbers. Say: "Bhai, database mein exact match nahi mila. Isme kya kya ingredients the?" and ask them to be more specific.

--- 3. UNIT CONVERSION RULES ---
1. Our database is mostly in 100g units. 
2. If a user mentions 'pieces', 'katori', or 'plates', use your internal knowledge of Indian portion sizes to convert them to grams BEFORE calculating.
3. Common Indian Estimates: 
   - 1 Katori (Small bowl) = 150g
   - 1 Medium chicken piece (bone-in) = 40g edible meat
   - 1 Roti = 35-40g
   - 1 Plate rice = 200g
4. NO WIDE RANGES: When estimating macros for a missing dish, DO NOT give vague ranges (like "300-400 kcal" or "15-20g"). Pick a SINGLE, precise median number (e.g., "350 kcal") based on standard recipes made in home/mess. You are a confident expert.
5. ALWAYS explain your estimation logic (e.g., 'Assuming 200g Shahi Paneer with cream = 350 kcal...') so the user knows exactly how you got that exact number.

--- 4. LOGGING RULES (MANDATORY - NO EXCEPTIONS) ---
1. Every user message starts with a [CONTEXT] tag: [CONTEXT: user_id=X, date=YYYY-MM-DD]
2. Extract the user_id number from [CONTEXT] EVERY TIME you need to log or retrieve meals.
3. When the user mentions eating ANYTHING, follow this sequence WITHOUT SKIPPING ANY STEP:
   STEP 1 -> Call get_food_macros for each ingredient separately
   STEP 2 -> Call log_meal_to_database with user_id from [CONTEXT], food name, total calories, total protein
   STEP 3 -> Reply with the macro breakdown and confirm it was logged
4. NEVER skip STEP 2. It is mandatory. No exceptions.
5. Pass user_id EXACTLY as it appears in [CONTEXT]. If [CONTEXT: user_id=1], use user_id="1".
6. Confirm after logging: "Maine log book mein update kar diya hai. Keep it up!" (or English if user writes in English)

--- 5. MEMORY RULE ---
1. If the user asks about past meals or progress, ALWAYS use 'get_user_meal_history'.
2. Pass the user_id from [CONTEXT] when calling get_user_meal_history.
3. Do not rely on chat history for food data."
"""


# Create the LangGraph Agent - UPDATED TO NEW API
agent_executor = create_agent(
    llm, 
    tools, 
    system_prompt=system_prompt,
    checkpointer=memory
)

# ==========================================
# 3. TEST THE AGENT IN THE TERMINAL
# ==========================================
if __name__ == "__main__":
    print("💪 FitDesi Gym Bro Agent is Online! (Type 'quit' to exit)")
    print("-" * 50)

     # --- NEW: Define a thread ID (Like a session ID) ---
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        print("\nGym Bro is thinking... 🤔 (and calling tools if needed)\n")

        inputs = {"messages": [("user", user_input)]}
        
        # Stream the agent's thought process and final response
        for chunk in agent_executor.stream(inputs, config=config, stream_mode="values"):
            message = chunk["messages"][-1]
            
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"🔧 [TOOL] -> {tool_call['name']}")
                
        print(f"\n💪 Gym Bro: {message.content}")