from stock_predictor.fastapi_module.dependencies import app, metadata, engine
from stock_predictor.fastapi_module.routers import (
    tickers_router,
    predicted_trades_router,
    stock_data_router,
)


metadata.create_all(bind=engine)


app.include_router(tickers_router.router)
app.include_router(predicted_trades_router.router)
app.include_router(stock_data_router.router)


@app.get("/")
async def root():
    return {"message": "Stock Predictor API"}
