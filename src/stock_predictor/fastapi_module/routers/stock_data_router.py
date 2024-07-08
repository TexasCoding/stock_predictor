from typing import List

from stock_predictor.global_settings import logger
from fastapi import APIRouter, HTTPException, Path, status

from stock_predictor.fastapi_module.dependencies import db_dependency
from stock_predictor.fastapi_module.models.stock_data_model import (
    StockData,
    StockDataRequest,
    StockDataResponse,
)

router = APIRouter()


######################
# GET /stock_data
######################
@router.get(
    "/stock_data",
    response_model=list[StockDataResponse],
    status_code=status.HTTP_200_OK,
)
async def read_all_stock_data(db: db_dependency):
    """
    Reads all stock data from the database.

    Args:
        db (db_dependency): The database dependency.

    Returns:
        List[StockData]: A list of all stock data in the database.
    """
    return db.query(StockData).all()


# ##########################
# # GET /stock_data/{stock_data_id}
# ##########################
# @router.get(
#     "/stock_data/{stock_data_id}",
#     response_model=StockDataResponse,
#     status_code=status.HTTP_200_OK,
# )
# async def get_stock_data_by_id(
#     db: db_dependency, stock_data_id: int = Path(..., title="Stock Data ID", ge=1)
# ):
#     """
#     Retrieve stock data by its ID.

#     Args:
#         db (db_dependency): The database dependency.
#         stock_data_id (int): The ID of the stock data to retrieve.

#     Returns:
#         StockData: The stock data object.

#     Raises:
#         HTTPException: If the stock data with the specified ID is not found.
#     """
#     stock_data = db.query(StockData).filter(StockData.id == stock_data_id).all()
#     if stock_data is None:
#         logger.warning(f"Stock data with ID {stock_data_id} not found.")
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND, detail="Stock data not found"
#         )
#     return stock_data


##########################
# GET /stock_data/{stock_symbol}
##########################
@router.get(
    "/stock_data/{stock_symbol}",
    response_model=list[StockDataResponse],
    status_code=status.HTTP_200_OK,
)
async def get_stock_data_by_symbol(
    db: db_dependency,
    stock_symbol: str = Path(..., title="Stock Symbol"),
    limit: int = 1095,
):
    """
    Retrieve stock data by its symbol.

    Args:
        db (db_dependency): The database dependency.
        stock_symbol (str): The symbol of the stock data to retrieve.

    Returns:
        StockData: The stock data object.

    Raises:
        HTTPException: If the stock data with the specified symbol is not found.
    """
    stock_data = (
        db.query(StockData)
        .order_by(StockData.date.desc())
        .filter(StockData.symbol == stock_symbol)
        .limit(limit)
        .all()
    )
    if stock_data is None:
        logger.warning(f"Stock data with symbol {stock_symbol} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Stock data not found"
        )
    return stock_data


######################
# POST /stock_data
######################
@router.post("/stock_data", status_code=status.HTTP_201_CREATED)
async def create_stock_data(
    db: db_dependency, stock_data_request: List[StockDataRequest]
):
    """
    Create new stock data in the database.

    Args:
        db (db_dependency): The database dependency.
        stock_data_request (StockDataRequest): The request object containing the stock data.

    Returns:
        StockData: The created stock data object.

    Raises:
        HTTPException: If there is an error creating the stock data.
    """
    try:
        objects = []
        for stock_data in stock_data_request:
            objects.append(StockData(**stock_data.model_dump()))
        # stock_data = StockData(**stock_data_request.model_dump())
        # db.add(stock_data)
        db.bulk_save_objects(objects)
        db.commit()
        # db.refresh(stock_data)  # Ensure the created stock data is returned with ID
        return
    except Exception as e:
        logger.error(f"Error creating stock data: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error creating stock data: {e}",
        )


###########################
# PUT /stock_data/{stock_data_id}
###########################
@router.put("/stock_data/{stock_data_id}", status_code=status.HTTP_204_NO_CONTENT)
async def update_stock_data(
    db: db_dependency,
    stock_data_request: StockDataRequest,
    stock_data_id: int = Path(..., title="Stock Data ID", ge=1),
):
    """
    Update stock data in the database.

    Args:
        db (db_dependency): The database dependency.
        stock_data_request (StockDataRequest): The updated stock data information.
        stock_data_id (int, optional): The ID of the stock data to update. Defaults to Path(..., title="Stock Data ID", ge=1).

    Raises:
        HTTPException: If the stock data with the given ID is not found.

    Returns:
        None
    """
    stock_data = db.query(StockData).filter(StockData.id == stock_data_id).first()
    if stock_data is None:
        logger.warning(f"Stock data with ID {stock_data_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Stock data not found"
        )
    for var, value in vars(stock_data_request).items():
        setattr(stock_data, var, value) if value else None
    db.commit()
    logger.info(f"Stock data with ID {stock_data_id} updated.")


##############################
# DELETE /stock_data/{stock_data_id}
##############################
@router.delete("/stock_data/{stock_data_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_stock_data(
    db: db_dependency, stock_data_id: int = Path(..., title="Stock Data ID", ge=1)
):
    """
    Delete stock data from the database.

    Args:
        db (db_dependency): The database dependency.
        stock_data_id (int): The ID of the stock data to be deleted.

    Raises:
        HTTPException: If the stock data with the specified ID is not found.

    Returns:
        None
    """
    stock_data = db.query(StockData).filter(StockData.id == stock_data_id).first()
    if stock_data is None:
        logger.warning(f"Stock data with ID {stock_data_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Stock data not found"
        )
    db.delete(stock_data)
    db.commit()
    logger.info(f"Stock data with ID {stock_data_id} deleted.")
