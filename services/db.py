# services/db.py
import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./app_data.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


class Annotation(Base):
    __tablename__ = "annotations"
    id = Column(Integer, primary_key=True, index=True)
    row_index = Column(Integer, index=True)
    text = Column(Text)
    label = Column(String(200))
    annotator = Column(String(200))
    note = Column(Text)
    auto = Column(Boolean, default=False)


class Task(Base):
    """
    Represents an annotation task / project owned by a user.
    Fields storing structured data are kept as JSON-text in SQLite for portability.
    """
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    owner = Column(String(200), index=True)           # simple owner identifier (username/email)
    name = Column(String(400))
    description = Column(Text)
    task_type = Column(String(50))                    # "multiclass" or "multilabel"
    selected_columns = Column(Text)                   # JSON list as string
    codebook_struct = Column(Text)                    # JSON list as string
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


# -------------------------
# Task helper functions
# -------------------------
def create_task(owner: str, name: str = "Untitled task", description: str = "", task_type: str = "multiclass"):
    db = SessionLocal()
    try:
        t = Task(owner=owner, name=name, description=description, task_type=task_type,
                 selected_columns=json.dumps([]), codebook_struct=json.dumps([]))
        db.add(t)
        db.commit()
        db.refresh(t)
        return t
    finally:
        db.close()


def get_tasks_for_user(owner: str):
    db = SessionLocal()
    try:
        rows = db.query(Task).filter(Task.owner == owner).order_by(Task.created_at.desc()).all()
        return rows
    finally:
        db.close()


def get_task(task_id: int):
    db = SessionLocal()
    try:
        return db.query(Task).filter(Task.id == task_id).first()
    finally:
        db.close()


def update_task(task_id: int, **kwargs):
    """
    kwargs may include: name, description, task_type, selected_columns (list), codebook_struct (list)
    """
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return None
        if "name" in kwargs:
            task.name = kwargs["name"]
        if "description" in kwargs:
            task.description = kwargs["description"]
        if "task_type" in kwargs:
            task.task_type = kwargs["task_type"]
        if "selected_columns" in kwargs:
            task.selected_columns = json.dumps(kwargs["selected_columns"])
        if "codebook_struct" in kwargs:
            task.codebook_struct = json.dumps(kwargs["codebook_struct"])
        db.add(task)
        db.commit()
        db.refresh(task)
        return task
    finally:
        db.close()
