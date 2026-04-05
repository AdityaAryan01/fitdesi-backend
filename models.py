from sqlalchemy import Column, DateTime, Integer, String, Float, ForeignKey, Date, Text
from sqlalchemy.orm import relationship
from database import Base
import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    weight_kg = Column(Float)
    height_cm = Column(Float)
    goal = Column(String) # e.g., "cut", "bulk", "maintenance"
    diet_type = Column(String) # e.g., "veg", "non-veg", "eggetarian"
    target_calories = Column(Integer)
    target_protein = Column(Integer)

    # Relationship to logs
    logs = relationship("DailyLog", back_populates="user")

class FoodItem(Base):
    __tablename__ = "food_items"

    id = Column(Integer, primary_key=True, index=True)
    item_name = Column(String, index=True)
    serving_size = Column(String) # e.g., "1 katori", "100g", "1 piece"
    calories = Column(Float)
    protein = Column(Float)
    carbs = Column(Float)
    fat = Column(Float)

class DailyLog(Base):
    __tablename__ = "daily_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    date = Column(Date, default=datetime.date.today)
    food_name = Column(String) # What the user ate
    calories = Column(Float)
    protein = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    user = relationship("User", back_populates="logs")
    
class ChatThread(Base):
    __tablename__ = "chat_threads"
    id = Column(String, primary_key=True) # thread_id
    user_id = Column(String, ForeignKey("users.id"))
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

# ✅ NEW: Persists every human/bot message per thread in SQLite
# This replaces MemorySaver (RAM-only) for history display in the UI.
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id         = Column(Integer, primary_key=True, index=True)
    thread_id  = Column(String, ForeignKey("chat_threads.id"), index=True)
    role       = Column(String)   # "user" or "bot"
    content    = Column(Text)     # full message text
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))