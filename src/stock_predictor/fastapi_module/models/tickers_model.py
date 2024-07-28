import datetime
import pendulum
from pydantic import BaseModel, Field
from stock_predictor.fastapi_module.database import Base
from sqlalchemy import Column, Integer, String, Float, DateTime


class Ticker(Base):
    __tablename__ = "tickers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    symbol = Column(String, index=True, unique=True)
    image = Column(String)
    market_cap = Column(Integer)
    gross_margin_pct = Column(Float)
    net_margin_pct = Column(Float)
    trailing_pe = Column(Float)
    piotroski_score = Column(Integer)
    industry = Column(String)
    sector = Column(String)
    created_at = Column(DateTime, default=pendulum.now(tz="America/New_York"))
    updated_at = Column(
        DateTime,
        default=pendulum.now(tz="America/New_York"),
        onupdate=pendulum.now(tz="America/New_York"),
    )


class TickerRequest(BaseModel):
    name: str = Field(..., title="Ticker name", max_length=100)
    symbol: str = Field(..., title="Ticker symbol", max_length=10)
    image: str = Field(..., title="Image URL")
    market_cap: int = Field(..., title="Market capitalization")
    gross_margin_pct: float = Field(..., title="Gross margin percentage")
    net_margin_pct: float = Field(..., title="Net margin percentage")
    trailing_pe: float = Field(..., title="Trailing PE ratio")
    piotroski_score: int = Field(..., title="Piotroski score")
    industry: str = Field(..., title="Industry", max_length=100)
    sector: str = Field(..., title="Sector", max_length=100)


class TickerResponse(BaseModel):
    id: int
    name: str
    symbol: str
    image: str
    market_cap: int
    gross_margin_pct: float
    net_margin_pct: float
    trailing_pe: float
    piotroski_score: int
    industry: str
    sector: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        from_attributes = True
