import os
from dotenv import load_dotenv
load_dotenv(override=True)
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Database URL from environment or fallback to SQLite
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/fitdesi.db")

if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    # connect_args is needed only for SQLite to allow multiple threads
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session in FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()