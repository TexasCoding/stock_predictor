# src/stock_trader/fastapi_module/database.py
# This module contains the database configuration for the stock_trader project.
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from stock_predictor.global_settings import DATA_DIR, DATABASE_FILE

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATA_DIR}{DATABASE_FILE}.db"

# Create a database engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
# Create a session object
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Create a base class for declarative class definitions
Base = declarative_base()
metadata = Base.metadata
