# main.py
from datetime import date, datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
import uuid
from pydantic import BaseModel

from database import SessionLocal, engine, Base
import models
from agent import agent_executor, present_response, generate_thread_title
from auth import verify_firebase_token          # ✅ Firebase token verifier

from langchain_core.messages import HumanMessage

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
    user_id: str
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    thread_title: str | None = None 

class UserCreate(BaseModel):
    id: str                     # ✅ Firebase UID passed explicitly
    name: str
    age: int
    weight_kg: float
    height_cm: float
    gender: str = "male"
    activity_level: str = "moderate"
    goal: str
    diet_type: str
    target_calories: int = 2000
    target_protein: int  = 120

class UserUpdate(BaseModel):
    name: str
    age: int
    weight_kg: float
    height_cm: float
    gender: str = "male"
    activity_level: str = "moderate"
    goal: str
    diet_type: str
    target_calories: int
    target_protein: int

class ThreadRenameRequest(BaseModel):
    title: str

# ── Routes ───────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "FitDesi Backend is LIVE! 🚀"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_gym_bro(
    request: ChatRequest,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    if token_data["uid"] != request.user_id:
        raise HTTPException(status_code=403, detail="User ID mismatch")

    try:
        config = {"configurable": {"thread_id": request.thread_id}, "recursion_limit": 50}
        
        user = db.query(models.User).filter(models.User.id == request.user_id).first()
        context_str = f"uid={request.user_id}, date={date.today()}"
        if user:
            context_str += f", diet={user.diet_type}, wt={user.weight_kg}kg, goal={user.goal}, cal={user.target_calories}, pro={user.target_protein}g"

        inputs = {
            "messages": [
                HumanMessage(content=f"[CONTEXT: {context_str}]\n{request.message}")
            ]
        }
        
        # 1. CORE AGENT (Heavy lifting & Math)
        result = agent_executor.invoke(inputs, config=config)
        raw_message = result["messages"][-1].content

        # 2. PRESENTER AGENT (Formatting)
        final_message = present_response(raw_message)

        # 3. Persist to SQLite
        db.add(models.ChatMessage(thread_id=request.thread_id, role="user", content=request.message))
        db.add(models.ChatMessage(thread_id=request.thread_id, role="bot",  content=final_message))
        db.commit()

        # 4. AUTO-TITLING LOGIC
        new_title = None
        thread = db.query(models.ChatThread).filter(models.ChatThread.id == request.thread_id).first()
        
        if thread and thread.title in ("New Conversation", "New Chat", "", None):
            # Only title on the first message exchange (user + bot = 2 messages in DB)
            msg_count = db.query(models.ChatMessage).filter(models.ChatMessage.thread_id == request.thread_id).count()
            if msg_count <= 2: 
                new_title = generate_thread_title(request.message)
                thread.title = new_title
                db.commit()
                print(f"📝 Auto-Titled Thread: {new_title}")

        return {"reply": final_message, "thread_title": new_title}

    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/{user_id}/progress")
def get_user_progress(
    user_id: str,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    if token_data["uid"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    today = date.today()
    logs  = db.query(models.DailyLog).filter(
        models.DailyLog.user_id == user_id,
        models.DailyLog.date == today
    ).all()
    return {
        "date":           str(today),
        "total_calories": sum(l.calories for l in logs),
        "total_protein":  sum(l.protein  for l in logs),
        "meals": [{"name": l.food_name, "kcal": l.calories, "protein": l.protein} for l in logs],
    }


# ── NEW: Weekly progress endpoint ────────────────────────────────
@app.get("/api/user/{user_id}/weekly")
def get_weekly_progress(
    user_id: str,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    """
    Returns last 7 days of macro logs grouped by date.
    Each day has: date, day_label (Mon/Tue/...), total_calories, total_protein.
    Days with no logs default to 0 — so the chart always shows a full 7-day window.
    """
    if token_data["uid"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    today = date.today()
    start = today - timedelta(days=6)  # 7-day window inclusive of today

    # Aggregate per day from DB
    rows = (
        db.query(
            models.DailyLog.date,
            func.sum(models.DailyLog.calories).label("total_calories"),
            func.sum(models.DailyLog.protein).label("total_protein"),
        )
        .filter(
            models.DailyLog.user_id == user_id,
            models.DailyLog.date >= start,
            models.DailyLog.date <= today,
        )
        .group_by(models.DailyLog.date)
        .all()
    )

    # Build a dict keyed by date string for O(1) lookup
    logged = {str(row.date): {"cal": round(row.total_calories, 1), "protein": round(row.total_protein, 1)} for row in rows}

    # Fill all 7 days — missing days get 0
    result = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        d_str = str(d)
        result.append({
            "date": d_str,
            "day": d.strftime("%a"),          # "Mon", "Tue", ...
            "cal": logged.get(d_str, {}).get("cal", 0),
            "protein": logged.get(d_str, {}).get("protein", 0),
        })

    return result


# ✅ NEW: Fetch single user profile — for returning users on login
@app.get("/api/user/{user_id}")
def get_user(
    user_id: str,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    if token_data["uid"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user.id, "name": user.name, "goal": user.goal,
        "diet_type": user.diet_type, "target_calories": user.target_calories,
        "target_protein": user.target_protein, "weight_kg": user.weight_kg,
        "height_cm": user.height_cm, "age": user.age,
        "gender": user.gender, "activity_level": user.activity_level,
    }


@app.post("/api/user")
def create_user(
    data: UserCreate,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    if token_data["uid"] != data.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    existing = db.query(models.User).filter(models.User.id == data.id).first()
    if existing:
        return {
            "id": existing.id, "name": existing.name, "goal": existing.goal,
            "diet_type": existing.diet_type, "target_calories": existing.target_calories,
            "target_protein": existing.target_protein, "weight_kg": existing.weight_kg,
            "height_cm": existing.height_cm, "age": existing.age,
            "gender": existing.gender, "activity_level": existing.activity_level,
        }

    user = models.User(
        id=data.id, name=data.name, age=data.age,
        weight_kg=data.weight_kg, height_cm=data.height_cm,
        gender=data.gender, activity_level=data.activity_level,
        goal=data.goal, diet_type=data.diet_type,
        target_calories=data.target_calories, target_protein=data.target_protein,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "id": user.id, "name": user.name, "goal": user.goal,
        "diet_type": user.diet_type, "target_calories": user.target_calories,
        "target_protein": user.target_protein, "weight_kg": user.weight_kg,
        "height_cm": user.height_cm, "age": user.age,
        "gender": user.gender, "activity_level": user.activity_level,
    }

@app.put("/api/user/{user_id}")
def update_user_profile(
    user_id: str,
    data: UserUpdate,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    if token_data["uid"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.name = data.name
    user.age = data.age
    user.weight_kg = data.weight_kg
    user.height_cm = data.height_cm
    user.gender = data.gender
    user.activity_level = data.activity_level
    user.goal = data.goal
    user.diet_type = data.diet_type
    user.target_calories = data.target_calories
    user.target_protein = data.target_protein

    db.commit()
    db.refresh(user)
    return {
        "id": user.id, "name": user.name, "goal": user.goal,
        "diet_type": user.diet_type, "target_calories": user.target_calories,
        "target_protein": user.target_protein, "weight_kg": user.weight_kg,
        "height_cm": user.height_cm, "age": user.age,
        "gender": user.gender, "activity_level": user.activity_level,
    }
@app.get("/api/user/{user_id}/logs")
def get_user_logs(
    user_id: str,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    if token_data["uid"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    today = date.today()
    logs  = db.query(models.DailyLog).filter(
        models.DailyLog.user_id == user_id,
        models.DailyLog.date == today
    ).order_by(models.DailyLog.id.desc()).all()
    return [
        {"id": l.id, "food_name": l.food_name, "calories": l.calories, "protein": l.protein, "date": str(l.date)}
        for l in logs
    ]


@app.delete("/api/user/{user_id}/logs/{log_id}")
def delete_user_log(
    user_id: str, log_id: int,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    if token_data["uid"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    db.query(models.DailyLog).filter(
        models.DailyLog.id == log_id,
        models.DailyLog.user_id == user_id
    ).delete()
    db.commit()
    return {"status": "deleted"}


# ── Thread Routes ────────────────────────────────────────────────

@app.get("/api/user/{user_id}/threads")
def get_user_threads(
    user_id: str,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    if token_data["uid"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return db.query(models.ChatThread).filter(
        models.ChatThread.user_id == user_id
    ).order_by(models.ChatThread.created_at.desc()).all()


@app.post("/api/user/{user_id}/threads")
def create_new_thread(
    user_id: str,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    if token_data["uid"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    new_id = f"thread_{uuid.uuid4().hex[:8]}"
    t = models.ChatThread(id=new_id, user_id=user_id, title="New Conversation", created_at=datetime.utcnow())
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@app.delete("/api/user/{user_id}/threads/{thread_id}")
def delete_thread(
    user_id: str, thread_id: str,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    if token_data["uid"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    db.query(models.ChatMessage).filter(models.ChatMessage.thread_id == thread_id).delete()
    db.query(models.ChatThread).filter(
        models.ChatThread.id == thread_id,
        models.ChatThread.user_id == user_id
    ).delete()
    db.commit()
    return {"status": "deleted"}


@app.put("/api/user/{user_id}/threads/{thread_id}")
def rename_thread(
    user_id: str, thread_id: str,
    request: ThreadRenameRequest,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    if token_data["uid"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    t = db.query(models.ChatThread).filter(
        models.ChatThread.id == thread_id,
        models.ChatThread.user_id == user_id
    ).first()
    if t:
        t.title = request.title
        db.commit()
        return {"status": "renamed", "title": t.title}
    raise HTTPException(status_code=404, detail="Thread not found")


@app.get("/api/chat/history/{thread_id}")
def get_chat_history(
    thread_id: str,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_firebase_token),
):
    try:

        thread = db.query(models.ChatThread).filter(
            models.ChatThread.id == thread_id,
            models.ChatThread.user_id == token_data["uid"]  # ownership check
        ).first()
        if not thread:
            raise HTTPException(status_code=403, detail="Forbidden")

        msgs = db.query(models.ChatMessage).filter(
            models.ChatMessage.thread_id == thread_id
        ).order_by(models.ChatMessage.id.asc()).all()
        return [{"role": m.role, "content": m.content} for m in msgs]
    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        return []