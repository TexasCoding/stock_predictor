from typing import Annotated

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from stock_predictor.fastapi_module.database import metadata
from stock_predictor.fastapi_module.database import SessionLocal, engine


app = FastAPI()

metadata = metadata
engine = engine


######################
# Dependency Injection
######################
def get_db():
    """
    Returns a database session.

    This function is used to get a database session for performing database operations.
    The session is created using the `SessionLocal` object and is closed after the operations are completed.

    Returns:
        SessionLocal: The database session.

    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Dependency for database session
db_dependency = Annotated[Session, Depends(get_db)]
