from datetime import date, datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uuid
from pydantic import BaseModel


from database import SessionLocal, engine, Base
import models
from agent import agent_executor

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
    user_id: str
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

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

@app.get("/")
def health_check():
    return {"status": "FitDesi Backend is LIVE! 🚀"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_gym_bro(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        config = {
            "configurable": {"thread_id": request.thread_id},
            "recursion_limit": 50
        }

        inputs = {
            "messages": [
                HumanMessage(content=(
                    f"[CONTEXT: user_id={request.user_id}, date={date.today()}]\n"
                    f"{request.message}"
                ))
            ]
        }

        result = agent_executor.invoke(inputs, config=config)
        final_message = result["messages"][-1].content

        # ✅ Save both sides of the conversation to SQLite so history
        # survives server restarts (MemorySaver is RAM-only).
        db.add(models.ChatMessage(
            thread_id=request.thread_id,
            role="user",
            content=request.message           # store the clean message, not the [CONTEXT] prefix
        ))
        db.add(models.ChatMessage(
            thread_id=request.thread_id,
            role="bot",
            content=final_message
        ))
        db.commit()

        return {"reply": final_message}

    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/{user_id}/progress")
def get_user_progress(user_id: str, db: Session = Depends(get_db)):
    today = date.today()
    logs = db.query(models.DailyLog).filter(
        models.DailyLog.user_id == user_id,
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
def create_user(data: UserCreate, db: Session = Depends(get_db)):
    user = models.User(
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


@app.get("/api/user/{user_id}/logs")
def get_user_logs(user_id: str, db: Session = Depends(get_db)):
    today = date.today()
    logs = (
        db.query(models.DailyLog)
        .filter(
            models.DailyLog.user_id == user_id,
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
        }
        for log in logs
    ]

# Schema for renaming a thread
class ThreadRenameRequest(BaseModel):
    title: str

# 1. FETCH ALL THREADS
@app.get("/api/user/{user_id}/threads")
def get_user_threads(user_id: str, db: Session = Depends(get_db)):
    return db.query(models.ChatThread).filter(
        models.ChatThread.user_id == user_id
    ).order_by(models.ChatThread.created_at.desc()).all()

# 2. CREATE A NEW THREAD
@app.post("/api/user/{user_id}/threads")
def create_new_thread(user_id: str, db: Session = Depends(get_db)):
    new_id = f"thread_{uuid.uuid4().hex[:8]}"
    new_thread = models.ChatThread(
        id=new_id, 
        user_id=user_id, 
        title="New Conversation",
        created_at=datetime.utcnow()
    )
    db.add(new_thread)
    db.commit()
    db.refresh(new_thread)
    return new_thread

# 3. DELETE A THREAD
@app.delete("/api/user/{user_id}/threads/{thread_id}")
def delete_thread(user_id: str, thread_id: str, db: Session = Depends(get_db)):
    # ✅ Also delete all messages belonging to this thread
    db.query(models.ChatMessage).filter(
        models.ChatMessage.thread_id == thread_id
    ).delete()
    db.query(models.ChatThread).filter(
        models.ChatThread.id == thread_id, 
        models.ChatThread.user_id == user_id
    ).delete()
    db.commit()
    return {"status": "deleted"}

# 4. RENAME A THREAD
@app.put("/api/user/{user_id}/threads/{thread_id}")
def rename_thread(user_id: str, thread_id: str, request: ThreadRenameRequest, db: Session = Depends(get_db)):
    thread = db.query(models.ChatThread).filter(
        models.ChatThread.id == thread_id, 
        models.ChatThread.user_id == user_id
    ).first()
    
    if thread:
        thread.title = request.title
        db.commit()
        return {"status": "renamed", "title": thread.title}
    raise HTTPException(status_code=404, detail="Thread not found")


# ✅ FIXED: Now reads from SQLite instead of MemorySaver (which resets on restart)
@app.get("/api/chat/history/{thread_id}")
def get_chat_history(thread_id: str, db: Session = Depends(get_db)):
    try:
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