import datetime
from pydantic import BaseModel, Field
from stock_predictor.fastapi_module.database import Base
from sqlalchemy import Column, Integer, String, Float, DateTime, UniqueConstraint


class StockData(Base):
    __tablename__ = "stock_data"
    __table_args__ = (UniqueConstraint("symbol", "date", name="symbol_date_unique"),)

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    vwap = Column(Float)


class StockDataRequest(BaseModel):
    symbol: str = Field(..., title="Stock symbol", max_length=10)
    date: datetime.datetime = Field(..., title="Date")
    open: float = Field(..., title="Opening price")
    high: float = Field(..., title="Highest price")
    low: float = Field(..., title="Lowest price")
    close: float = Field(..., title="Closing price")
    volume: int = Field(..., title="Volume")
    vwap: float = Field(..., title="Volume weighted average price")


class StockDataResponse(BaseModel):
    id: int
    symbol: str
    date: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float

    class Config:
        from_attributes = True


class StockDataQuery(BaseModel):
    symbol: str
