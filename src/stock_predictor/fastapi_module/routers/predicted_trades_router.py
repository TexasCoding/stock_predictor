from stock_predictor.fastapi_module.models.predicted_trades_model import (
    PredictedTrades,
    PredictedTradesRequest,
    PredictedTradesResponse,
)
from stock_predictor.global_settings import logger
from fastapi import APIRouter, HTTPException, Path, status

from stock_predictor.fastapi_module.dependencies import db_dependency

router = APIRouter()


######################
# GET /predicted_trades
######################
@router.get(
    "/predicted_trades",
    response_model=list[PredictedTradesResponse],
    status_code=status.HTTP_200_OK,
)
async def read_all_predicted_trades(db: db_dependency):
    """
    Reads all predicted trades from the database.

    Args:
        db (db_dependency): The database dependency.

    Returns:
        List[PredictedTrades]: A list of all predicted trades in the database.
    """
    return db.query(PredictedTrades).all()


######################
# GET /predicted_trades/{predicted_trade_id}
######################
@router.get(
    "/predicted_trades/{predicted_trade_id}",
    response_model=PredictedTradesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_predicted_trade_by_id(
    db: db_dependency,
    predicted_trade_id: int = Path(..., title="Predicted Trade ID", ge=1),
):
    """
    Retrieve a predicted trade by its ID.

    Args:
        db (db_dependency): The database dependency.
        predicted_trade_id (int): The ID of the predicted trade to retrieve.

    Returns:
        PredictedTrades: The predicted trade object.

    Raises:
        HTTPException: If the predicted trade with the specified ID is not found.
    """
    predicted_trade = (
        db.query(PredictedTrades)
        .filter(PredictedTrades.id == predicted_trade_id)
        .first()
    )
    if predicted_trade is None:
        logger.warning(f"Predicted trade with ID {predicted_trade_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Predicted trade not found"
        )
    return predicted_trade


######################
# POST /predicted_trades
######################
@router.post("/predicted_trades", status_code=status.HTTP_201_CREATED)
async def create_predicted_trade(
    db: db_dependency, predicted_trade_request: PredictedTradesRequest
):
    """
    Create a new predicted trade in the database.

    Args:
        db (db_dependency): The database dependency.
        predicted_trade_request (PredictedTradesRequest): The request object containing the predicted trade data.

    Returns:
        PredictedTrades: The created predicted trade object.

    Raises:
        HTTPException: If there is an error creating the predicted trade.
    """
    try:
        predicted_trade = PredictedTrades(**predicted_trade_request.model_dump())
        db.add(predicted_trade)
        db.commit()
        db.refresh(
            predicted_trade
        )  # Ensure the created predicted trade is returned with ID
        logger.info(
            f"Predicted trade {predicted_trade.symbol} created with ID {predicted_trade.id}."
        )
        return predicted_trade
    except Exception as e:
        logger.error(f"Error creating predicted trade: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error creating predicted trade: {e}",
        )


###########################
# PUT /predicted_trades/{predicted_trade_id}
###########################
@router.put(
    "/predicted_trades/{predicted_trade_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def update_predicted_trade(
    db: db_dependency,
    predicted_trade_request: PredictedTradesRequest,
    predicted_trade_id: int = Path(..., title="Predicted Trade ID", ge=1),
):
    """
    Update a predicted trade in the database.

    Args:
        db (db_dependency): The database dependency.
        predicted_trade_request (PredictedTradesRequest): The updated predicted trade information.
        predicted_trade_id (int, optional): The ID of the predicted trade to update. Defaults to Path(..., title="Predicted Trade ID", ge=1).

    Raises:
        HTTPException: If the predicted trade with the given ID is not found.

    Returns:
        None
    """
    predicted_trade = (
        db.query(PredictedTrades)
        .filter(PredictedTrades.id == predicted_trade_id)
        .first()
    )
    if predicted_trade is None:
        logger.warning(f"Predicted trade with ID {predicted_trade_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Predicted trade not found"
        )
    for var, value in vars(predicted_trade_request).items():
        setattr(predicted_trade, var, value) if value else None
    db.commit()
    logger.info(f"Predicted trade with ID {predicted_trade_id} updated.")


##############################
# DELETE /predicted_trades/{predicted_trade_id}
##############################
@router.delete(
    "/predicted_trades/{predicted_trade_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_predicted_trade(
    db: db_dependency,
    predicted_trade_id: int = Path(..., title="Predicted Trade ID", ge=1),
):
    """
    Delete a predicted trade from the database.

    Args:
        db (db_dependency): The database dependency.
        predicted_trade_id (int): The ID of the predicted trade to be deleted.

    Raises:
        HTTPException: If the predicted trade with the specified ID is not found.

    Returns:
        None
    """
    predicted_trade = (
        db.query(PredictedTrades)
        .filter(PredictedTrades.id == predicted_trade_id)
        .first()
    )
    if predicted_trade is None:
        logger.warning(f"Predicted trade with ID {predicted_trade_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Predicted trade not found"
        )
    db.delete(predicted_trade)
    db.commit()
    logger.info(f"Predicted trade with ID {predicted_trade_id} deleted.")
