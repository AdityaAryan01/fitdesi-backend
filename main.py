from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uuid
from pydantic import BaseModel


from database import SessionLocal, engine, Base
import models
from agent import agent_executor, detect_meal_intent, present_response, get_food_breakdown, build_system_prompt, memory, llm, tools, generate_thread_title

# LangChain message objects — same format agent.py uses internally
from langchain_core.messages import HumanMessage, SystemMessage

Base.metadata.create_all(bind=engine)

app = FastAPI(title="FitDesi AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── Pydantic Schemas ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    thread_title: str | None = None  # returned when thread is auto-titled on first message

class UserCreate(BaseModel):
    name: str
    age: int
    weight_kg: float
    height_cm: float
    goal: str
    diet_type: str
    target_calories: int = 2000
    target_protein: int  = 120

# ── Routes ───────────────────────────────────────────────────────

from auth import get_current_user_uid

@app.get("/")
def health_check():
    return {"status": "FitDesi Backend is LIVE! 🚀"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_gym_bro(request: ChatRequest, db: Session = Depends(get_db), uid: str = Depends(get_current_user_uid)):
    try:
        # ── STEP 0: Load user profile for personalization ─────────────────
        user_profile = None
        db_user = db.query(models.User).filter(models.User.id == uid).first()
        if db_user:
            user_profile = {
                "name":            db_user.name,
                "goal":            db_user.goal,
                "diet_type":       db_user.diet_type,
                "target_calories": db_user.target_calories,
                "target_protein":  db_user.target_protein,
                "weight_kg":       db_user.weight_kg,
            }
        diet_type = (user_profile or {}).get("diet_type", "veg") or "veg"
        print(f"[Pipeline] User profile loaded: name={user_profile.get('name') if user_profile else 'UNKNOWN'}, diet={diet_type}")

        # ── Build a personalized agent with this user's profile ───────────
        from langchain.agents import create_agent
        personalized_prompt = build_system_prompt(user_profile)
        personalized_agent = create_agent(
            llm,
            tools,
            system_prompt=personalized_prompt,
            checkpointer=memory,
        )

        # ── STEP 1: Intent Detection ─────────────────────────────────────
        is_meal_log = detect_meal_intent(request.message)
        print(f"[Pipeline] Intent detected — is_meal_log={is_meal_log}")

        config = {
            "configurable": {"thread_id": request.thread_id},
            "recursion_limit": 50
        }

        user_message_with_context = (
            f"[CONTEXT: user_id={uid}, date={date.today()}]\n"
            f"{request.message}"
        )

        # ── STEP 1b: Food Breakdown Agent (only when logging meals) ──────
        breakdown = None
        if is_meal_log:
            print(f"[Pipeline] Running food breakdown agent...")
            breakdown = get_food_breakdown(request.message, diet_type=diet_type)
            print(f"[Pipeline] Breakdown: {breakdown}")

        if not is_meal_log:
            messages = [
                SystemMessage(content=(
                    "IMPORTANT OVERRIDE: The user's message is NOT about logging food. "
                    "Do NOT call 'log_meal_to_database' under any circumstances. "
                    "Only answer questions or retrieve history if asked."
                )),
                HumanMessage(content=user_message_with_context)
            ]
        else:
            messages = [HumanMessage(content=user_message_with_context)]

        # ── STEP 2: Core Agent (personalized) ───────────────────────────
        result = personalized_agent.invoke({"messages": messages}, config=config)
        raw_response = result["messages"][-1].content
        print(f"[Pipeline] Raw agent response: {raw_response[:120]}...")

        # ── STEP 3: Presenter Agent (with diet enforcement) ──────────────
        final_message = present_response(request.message, raw_response, breakdown, user_profile=user_profile)
        print(f"[Pipeline] Presented response: {final_message[:120]}...")

        # Persist both sides to SQLite
        db.add(models.ChatMessage(
            thread_id=request.thread_id,
            role="user",
            content=request.message
        ))
        db.add(models.ChatMessage(
            thread_id=request.thread_id,
            role="bot",
            content=final_message
        ))
        db.commit()

        # ── Auto-title the thread on first message ──────────────────────────
        new_title = None
        thread = db.query(models.ChatThread).filter(
            models.ChatThread.id == request.thread_id
        ).first()
        if thread and thread.title in ("New Conversation", "", None):
            # Count messages — only title on the very first exchange
            msg_count = db.query(models.ChatMessage).filter(
                models.ChatMessage.thread_id == request.thread_id
            ).count()
            if msg_count <= 2:  # user + bot = 2 messages = first exchange
                new_title = generate_thread_title(request.message)
                thread.title = new_title
                db.commit()
                print(f"[Auto-Title] Thread {request.thread_id} titled: {new_title}")

        return {"reply": final_message, "thread_title": new_title}

    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/progress")
def get_user_progress(db: Session = Depends(get_db), uid: str = Depends(get_current_user_uid)):
    today = datetime.now(ZoneInfo('Asia/Kolkata')).date()
    logs = db.query(models.DailyLog).filter(
        models.DailyLog.user_id == uid,
        models.DailyLog.date == today
    ).all()

    return {
        "date":           str(today),
        "total_calories": sum(log.calories for log in logs),
        "total_protein":  sum(log.protein  for log in logs),
        "meals": [
            {"name": l.food_name, "kcal": l.calories, "protein": l.protein}
            for l in logs
        ]
    }


@app.post("/api/user")
def create_user(data: UserCreate, db: Session = Depends(get_db), uid: str = Depends(get_current_user_uid)):
    # Check if user already exists
    existing_user = db.query(models.User).filter(models.User.id == uid).first()
    if existing_user:
        return {
            "id":              existing_user.id,
            "name":            existing_user.name,
            "goal":            existing_user.goal,
            "diet_type":       existing_user.diet_type,
            "target_calories": existing_user.target_calories,
            "target_protein":  existing_user.target_protein,
            "weight_kg":       existing_user.weight_kg,
            "height_cm":       existing_user.height_cm,
            "age":             existing_user.age,
        }

    user = models.User(
        id=uid,
        name=data.name,
        age=data.age,
        weight_kg=data.weight_kg,
        height_cm=data.height_cm,
        goal=data.goal,
        diet_type=data.diet_type,
        target_calories=data.target_calories,
        target_protein=data.target_protein,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "id":              user.id,
        "name":            user.name,
        "goal":            user.goal,
        "diet_type":       user.diet_type,
        "target_calories": user.target_calories,
        "target_protein":  user.target_protein,
        "weight_kg":       user.weight_kg,
        "height_cm":       user.height_cm,
        "age":             user.age,
    }


@app.get("/api/user/me")
def get_current_user_profile(db: Session = Depends(get_db), uid: str = Depends(get_current_user_uid)):
    existing_user = db.query(models.User).filter(models.User.id == uid).first()
    if existing_user:
        return {
            "id":              existing_user.id,
            "name":            existing_user.name,
            "goal":            existing_user.goal,
            "diet_type":       existing_user.diet_type,
            "target_calories": existing_user.target_calories,
            "target_protein":  existing_user.target_protein,
            "weight_kg":       existing_user.weight_kg,
            "height_cm":       existing_user.height_cm,
            "age":             existing_user.age,
        }
    raise HTTPException(status_code=404, detail="User profile not found")


@app.get("/api/user/logs")
def get_user_logs(db: Session = Depends(get_db), uid: str = Depends(get_current_user_uid)):
    today = datetime.now(ZoneInfo('Asia/Kolkata')).date()
    logs = (
        db.query(models.DailyLog)
        .filter(
            models.DailyLog.user_id == uid,
            models.DailyLog.date == today
        )
        .order_by(models.DailyLog.id.desc())
        .all()
    )
    return [
        {
            "id":        log.id,
            "food_name": log.food_name,
            "calories":  log.calories,
            "protein":   log.protein,
            "date":      str(log.date),
            "created_at": log.created_at.isoformat() if getattr(log, 'created_at', None) else None,
        }
        for log in logs
    ]

@app.delete("/api/user/logs/{log_id}")
def delete_meal_log(log_id: int, db: Session = Depends(get_db), uid: str = Depends(get_current_user_uid)):
    log = db.query(models.DailyLog).filter(
        models.DailyLog.id == log_id,
        models.DailyLog.user_id == uid   # security: user can only delete their own
    ).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log entry not found")
    db.delete(log)
    db.commit()
    return {"status": "deleted", "id": log_id}

# Schema for renaming a thread
class ThreadRenameRequest(BaseModel):
    title: str

# 1. FETCH ALL THREADS
@app.get("/api/user/threads")
def get_user_threads(db: Session = Depends(get_db), uid: str = Depends(get_current_user_uid)):
    return db.query(models.ChatThread).filter(
        models.ChatThread.user_id == uid
    ).order_by(models.ChatThread.created_at.desc()).all()

# 2. CREATE A NEW THREAD
@app.post("/api/user/threads")
def create_new_thread(db: Session = Depends(get_db), uid: str = Depends(get_current_user_uid)):
    new_id = f"thread_{uuid.uuid4().hex[:8]}"
    new_thread = models.ChatThread(
        id=new_id, 
        user_id=uid, 
        title="New Conversation",
        created_at=datetime.utcnow()
    )
    db.add(new_thread)
    db.commit()
    db.refresh(new_thread)
    return new_thread

# 3. DELETE A THREAD
@app.delete("/api/user/threads/{thread_id}")
def delete_thread(thread_id: str, db: Session = Depends(get_db), uid: str = Depends(get_current_user_uid)):
    # ✅ Also delete all messages belonging to this thread
    db.query(models.ChatMessage).filter(
        models.ChatMessage.thread_id == thread_id
    ).delete()
    db.query(models.ChatThread).filter(
        models.ChatThread.id == thread_id, 
        models.ChatThread.user_id == uid
    ).delete()
    db.commit()
    return {"status": "deleted"}

# 4. RENAME A THREAD
@app.put("/api/user/threads/{thread_id}")
def rename_thread(thread_id: str, request: ThreadRenameRequest, db: Session = Depends(get_db), uid: str = Depends(get_current_user_uid)):
    thread = db.query(models.ChatThread).filter(
        models.ChatThread.id == thread_id, 
        models.ChatThread.user_id == uid
    ).first()
    
    if thread:
        thread.title = request.title
        db.commit()
        return {"status": "renamed", "title": thread.title}
    raise HTTPException(status_code=404, detail="Thread not found")


# ✅ FIXED: Now reads from SQLite instead of MemorySaver (which resets on restart)
@app.get("/api/chat/history/{thread_id}")
def get_chat_history(thread_id: str, db: Session = Depends(get_db), uid: str = Depends(get_current_user_uid)):
    try:
        # Secure endpoint: check if thread belongs to user
        thread = db.query(models.ChatThread).filter(
            models.ChatThread.id == thread_id,
            models.ChatThread.user_id == uid
        ).first()
        if not thread:
            raise HTTPException(status_code=403, detail="Not authorized to view this thread")

        messages = (
            db.query(models.ChatMessage)
            .filter(models.ChatMessage.thread_id == thread_id)
            .order_by(models.ChatMessage.id.asc())
            .all()
        )
        return [{"role": m.role, "content": m.content} for m in messages]
    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        return []