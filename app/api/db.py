# app/api/db.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# app/api/db.py
import os
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://app:app@postgres:5432/threatlogs"
)


# Motor y sesión
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Modelo de logs
class LogEntry(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source_ip = Column(String, index=True)
    dest_ip = Column(String)
    path = Column(String)
    status_code = Column(Integer)
    bytes = Column(Integer)
    response_time_ms = Column(Float)
    anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float)

# Crear tablas si no existen
def init_db():
    Base.metadata.create_all(bind=engine)
