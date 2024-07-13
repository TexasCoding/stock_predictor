from stock_predictor.global_settings import logger
from fastapi import APIRouter, HTTPException, Path, status

from stock_predictor.fastapi_module.dependencies import db_dependency
from stock_predictor.fastapi_module.models.tickers_model import (
    Ticker,
    TickerRequest,
    TickerResponse,
)

router = APIRouter()


######################
# GET /tickers
######################
@router.get(
    "/tickers", response_model=list[TickerResponse], status_code=status.HTTP_200_OK
)
async def read_all_tickers(db: db_dependency):
    """
    Reads all tickers from the database.

    Args:
        db (db_dependency): The database dependency.

    Returns:
        List[Ticker]: A list of all tickers in the database.
    """
    return db.query(Ticker).all()


##########################
# GET /tickers/{ticker_id}
##########################
@router.get(
    "/tickers/{ticker_id}",
    response_model=TickerResponse,
    status_code=status.HTTP_200_OK,
)
async def get_ticker_by_id(
    db: db_dependency, ticker_id: int = Path(..., title="Ticker ID", ge=1)
):
    """
    Retrieve a ticker by its ID.

    Args:
        db (db_dependency): The database dependency.
        ticker_id (int): The ID of the ticker to retrieve.

    Returns:
        Ticker: The ticker object.

    Raises:
        HTTPException: If the ticker with the specified ID is not found.
    """
    ticker = db.query(Ticker).filter(Ticker.id == ticker_id).first()
    if ticker is None:
        logger.warning(f"Ticker with ID {ticker_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Ticker not found"
        )
    return ticker


######################
# POST /tickers
######################
@router.post("/tickers", status_code=status.HTTP_201_CREATED)
async def create_ticker(db: db_dependency, ticker_request: TickerRequest):
    """
    Create a new ticker in the database.

    Args:
        db (db_dependency): The database dependency.
        ticker_request (TickerRequest): The request object containing the ticker data.

    Returns:
        Ticker: The created ticker object.

    Raises:
        HTTPException: If there is an error creating the ticker.
    """
    try:
        ticker = Ticker(**ticker_request.model_dump())
        db.add(ticker)
        db.commit()
        db.refresh(ticker)  # Ensure the created ticker is returned with ID
        logger.info(f"Ticker {ticker.name} created with ID {ticker.id}.")
        return ticker
    except Exception as e:
        logger.error(f"Error creating ticker: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error creating ticker: {e}",
        )


###########################
# PUT /tickers/{ticker_id}
###########################
@router.put("/tickers/{ticker_id}", status_code=status.HTTP_204_NO_CONTENT)
async def update_ticker(
    db: db_dependency,
    ticker_request: TickerRequest,
    ticker_id: int = Path(..., title="Ticker ID", ge=1),
):
    """
    Update a ticker in the database.

    Args:
        db (db_dependency): The database dependency.
        ticker_request (TickerRequest): The updated ticker information.
        ticker_id (int, optional): The ID of the ticker to update. Defaults to Path(..., title="Ticker ID", ge=1).

    Raises:
        HTTPException: If the ticker with the given ID is not found.

    Returns:
        None
    """
    ticker = db.query(Ticker).filter(Ticker.id == ticker_id).first()
    if ticker is None:
        logger.warning(f"Ticker with ID {ticker_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Ticker not found"
        )
    for var, value in vars(ticker_request).items():
        setattr(ticker, var, value) if value else None
    db.commit()
    logger.info(f"Ticker with ID {ticker_id} updated.")


##############################
# DELETE /tickers/{ticker_id}
##############################
@router.delete("/tickers/{ticker_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ticker(
    db: db_dependency, ticker_id: int = Path(..., title="Ticker ID", ge=1)
):
    """
    Delete a ticker from the database.

    Args:
        db (db_dependency): The database dependency.
        ticker_id (int): The ID of the ticker to be deleted.

    Raises:
        HTTPException: If the ticker with the specified ID is not found.

    Returns:
        None
    """
    ticker = db.query(Ticker).filter(Ticker.id == ticker_id).first()
    if ticker is None:
        logger.warning(f"Ticker with ID {ticker_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Ticker not found"
        )
    db.delete(ticker)
    db.commit()
    logger.info(f"Ticker with ID {ticker_id} deleted.")
