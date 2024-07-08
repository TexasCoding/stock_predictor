import datetime
import pendulum
from pydantic import BaseModel, Field
from stock_predictor.fastapi_module.database import Base
from sqlalchemy import Column, Integer, String, Float, DateTime


class PredictedTrades(Base):
    __tablename__ = "predicted_trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    open_price = Column(Float)
    take_price = Column(Float)
    created_at = Column(DateTime, default=pendulum.now(tz="America/New_York"))


class PredictedTradesRequest(BaseModel):
    symbol: str = Field(..., title="Stock symbol", max_length=10)
    open_price: float = Field(..., title="Open price")
    take_price: float = Field(..., title="Take price")


class PredictedTradesResponse(BaseModel):
    id: int
    symbol: str
    open_price: float
    take_price: float
    created_at: datetime.datetime

    class Config:
        from_attributes = True
