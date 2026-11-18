"""Portfolio API endpoints."""

import logging
from datetime import datetime
from math import fsum
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from Backend.App.database.connection import get_db
from Backend.App.database.crud import (
    DatabaseOperationError,
    RecordNotFoundError,
    add_holding as create_database_holding,
    create_portfolio as create_database_portfolio,
    delete_holding as delete_database_holding,
    get_user_portfolios,
    update_holding as update_database_holding,
)
from Backend.App.database.models import Holding as DatabaseHolding
from Backend.App.database.models import Portfolio as DatabasePortfolio
from Backend.App.database.models import User
from Backend.App.portfolio_service import (
    PortfolioHolding,
    PortfolioService,
    StockPrice,
    get_portfolio_service,
)
from Backend.App.security import get_current_user
from Backend.App.stock_catalog import STOCK_CATALOG

logger = logging.getLogger(__name__)
router = APIRouter()


class HoldingCreateRequest(BaseModel):
    """Input required to create a portfolio holding."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Supported Indian stock symbol, for example RELIANCE",
    )
    portfolio_id: int = Field(..., gt=0, description="ID of the destination portfolio")
    quantity: int = Field(..., gt=0, description="Number of shares held")
    purchase_price: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        description="Purchase price per share in INR",
    )


class HoldingUpdateRequest(BaseModel):
    """Replacement quantity and purchase price for an existing holding."""

    model_config = ConfigDict(extra="forbid")

    quantity: int = Field(..., gt=0, description="Updated number of shares held")
    purchase_price: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        description="Updated purchase price per share in INR",
    )


class HoldingResponse(BaseModel):
    """Portfolio holding with investment and optional latest-price valuation."""

    holding_id: UUID
    symbol: str
    quantity: int
    purchase_price: float
    total_investment: float = Field(..., description="Quantity multiplied by purchase price")
    current_price: float | None = Field(
        ..., description="Latest available closing price in INR, or null if unavailable"
    )
    current_value: float | None = Field(
        ..., description="Quantity multiplied by current price, or null if unavailable"
    )
    profit_loss: float | None = Field(
        ..., description="Current value minus investment, or null if price is unavailable"
    )
    price_timestamp: datetime | None = Field(
        ..., description="Timestamp of the latest available closing price"
    )


class PortfolioResponse(BaseModel):
    """Portfolio holdings and aggregate investment valuation."""

    holdings: list[HoldingResponse]
    total_investment: float
    current_value: float | None = Field(
        ..., description="Aggregate value, or null if any holding lacks price data"
    )
    profit_loss: float | None = Field(
        ..., description="Aggregate profit/loss, or null if any holding lacks price data"
    )


class PortfolioCreateRequest(BaseModel):
    """Name for a portfolio owned by the authenticated user."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    portfolio_name: str = Field(..., min_length=1, max_length=150)


class PortfolioSummaryResponse(BaseModel):
    """Public portfolio identifier and name."""

    portfolio_id: int
    portfolio_name: str


def _holding_response(
    holding: PortfolioHolding, price: StockPrice | None
) -> HoldingResponse:
    investment = holding.quantity * holding.purchase_price
    current_value = holding.quantity * price.value if price is not None else None
    return HoldingResponse(
        holding_id=holding.holding_id,
        symbol=holding.symbol,
        quantity=holding.quantity,
        purchase_price=holding.purchase_price,
        total_investment=investment,
        current_price=price.value if price is not None else None,
        current_value=current_value,
        profit_loss=current_value - investment if current_value is not None else None,
        price_timestamp=price.timestamp if price is not None else None,
    )


def _portfolio_response(
    holdings: list[PortfolioHolding], service: PortfolioService
) -> PortfolioResponse:
    valued_holdings: list[HoldingResponse] = []
    for holding in holdings:
        try:
            price = service.get_latest_price(holding.symbol)
        except Exception:
            logger.exception("Price service failed for portfolio symbol %s", holding.symbol)
            price = None
        valued_holdings.append(_holding_response(holding, price))

    total_investment = fsum(item.total_investment for item in valued_holdings)
    if all(item.current_value is not None for item in valued_holdings):
        current_value: float | None = fsum(
            item.current_value for item in valued_holdings if item.current_value is not None
        )
        profit_loss: float | None = current_value - total_investment
    else:
        current_value = None
        profit_loss = None

    return PortfolioResponse(
        holdings=valued_holdings,
        total_investment=total_investment,
        current_value=current_value,
        profit_loss=profit_loss,
    )


def _owned_portfolio(
    db: Session, portfolio_id: int, user_id: int
) -> DatabasePortfolio | None:
    """Return a portfolio only when it belongs to the supplied user."""
    statement = select(DatabasePortfolio).where(
        DatabasePortfolio.portfolio_id == portfolio_id,
        DatabasePortfolio.user_id == user_id,
    )
    return db.scalar(statement)


def _owned_database_holding(
    db: Session, holding_id: UUID, user_id: int
) -> DatabaseHolding | None:
    """Resolve the integer database ID encoded in the API's UUID field."""
    database_id = holding_id.int
    if database_id > 2**63 - 1:
        return None
    statement = (
        select(DatabaseHolding)
        .join(DatabasePortfolio, DatabaseHolding.portfolio_id == DatabasePortfolio.portfolio_id)
        .where(
            DatabaseHolding.holding_id == database_id,
            DatabasePortfolio.user_id == user_id,
        )
    )
    return db.scalar(statement)


@router.post(
    "/portfolio",
    response_model=PortfolioSummaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a portfolio",
)
def create_portfolio(
    request: PortfolioCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> PortfolioSummaryResponse:
    """Create a portfolio for the authenticated user."""
    try:
        portfolio = create_database_portfolio(
            db,
            user_id=current_user.user_id,
            portfolio_name=request.portfolio_name,
        )
    except DatabaseOperationError as exc:
        raise _database_unavailable(exc) from exc
    except RecordNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user no longer exists.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return PortfolioSummaryResponse(
        portfolio_id=portfolio.portfolio_id,
        portfolio_name=portfolio.portfolio_name,
    )


@router.get(
    "/portfolio",
    response_model=list[PortfolioSummaryResponse],
    summary="List the authenticated user's portfolios",
)
def list_portfolios(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[PortfolioSummaryResponse]:
    """Return only portfolios owned by the authenticated user."""
    portfolios = get_user_portfolios(db, current_user.user_id)
    return [
        PortfolioSummaryResponse(
            portfolio_id=portfolio.portfolio_id,
            portfolio_name=portfolio.portfolio_name,
        )
        for portfolio in portfolios
    ]


def _to_api_holding(holding: DatabaseHolding) -> PortfolioHolding:
    """Adapt a persisted holding to the existing response representation."""
    return PortfolioHolding(
        holding_id=UUID(int=holding.holding_id),
        symbol=holding.stock_symbol,
        quantity=holding.quantity,
        purchase_price=float(holding.purchase_price),
    )


def _database_unavailable(exc: Exception) -> HTTPException:
    logger.error(
        "Portfolio database operation failed (%s).",
        type(exc).__name__,
    )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Portfolio storage is currently unavailable.",
    )


@router.post(
    "/portfolio/holdings",
    response_model=HoldingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a portfolio holding",
    description="Add a supported stock holding using its share quantity and purchase price.",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "The stock symbol is not supported."
        }
    },
)
def add_holding(
    request: HoldingCreateRequest,
    service: Annotated[PortfolioService, Depends(get_portfolio_service)],
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> HoldingResponse:
    """Create a holding and include valuation only when provider data is available."""
    try:
        if _owned_portfolio(db, request.portfolio_id, current_user.user_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio was not found.",
            )
        database_holding = create_database_holding(
            db,
            portfolio_id=request.portfolio_id,
            stock_symbol=request.symbol,
            quantity=request.quantity,
            purchase_price=request.purchase_price,
        )
        holding = _to_api_holding(database_holding)
    except HTTPException:
        raise
    except DatabaseOperationError as exc:
        raise _database_unavailable(exc) from exc
    except RecordNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        if request.symbol.strip().upper() not in STOCK_CATALOG:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock symbol '{request.symbol.strip().upper()}' is not supported.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Could not add portfolio holding")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not add the portfolio holding.",
        ) from exc

    try:
        price = service.get_latest_price(holding.symbol)
    except Exception:
        logger.exception("Price service failed for portfolio symbol %s", holding.symbol)
        price = None
    return _holding_response(holding, price)


@router.get(
    "/portfolio/holdings",
    response_model=PortfolioResponse,
    summary="View portfolio holdings",
    description=(
        "Return holdings from the authenticated user's selected portfolio, total "
        "investment and latest available provider-based valuation. Prices and "
        "profit/loss are null when unavailable."
    ),
)
def list_holdings(
    service: Annotated[PortfolioService, Depends(get_portfolio_service)],
    db: Annotated[Session, Depends(get_db)],
    portfolio_id: Annotated[int, Query(gt=0, description="Portfolio to retrieve")],
    current_user: Annotated[User, Depends(get_current_user)],
) -> PortfolioResponse:
    """List persisted holdings for one owned portfolio and calculate valuations."""
    if _owned_portfolio(
        db, portfolio_id, current_user.user_id
    ) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio was not found.",
        )
    statement = (
        select(DatabaseHolding)
        .where(DatabaseHolding.portfolio_id == portfolio_id)
        .order_by(DatabaseHolding.holding_id)
    )
    try:
        holdings = [_to_api_holding(item) for item in db.scalars(statement).all()]
    except DatabaseOperationError as exc:
        raise _database_unavailable(exc) from exc
    return _portfolio_response(holdings, service)


@router.put(
    "/portfolio/holdings/{holding_id}",
    response_model=HoldingResponse,
    summary="Update a portfolio holding",
    description="Replace the quantity and purchase price of an existing holding.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "The holding does not exist."}
    },
)
def update_holding(
    request: HoldingUpdateRequest,
    holding_id: Annotated[UUID, Path(description="ID returned when the holding was added")],
    service: Annotated[PortfolioService, Depends(get_portfolio_service)],
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> HoldingResponse:
    """Update an existing holding and return its latest available valuation."""
    try:
        database_holding = _owned_database_holding(
            db, holding_id, current_user.user_id
        )
        if database_holding is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio holding was not found.",
            )
        database_holding = update_database_holding(
            db,
            database_holding.holding_id,
            quantity=request.quantity,
            purchase_price=request.purchase_price,
        )
        holding = _to_api_holding(database_holding)
    except HTTPException:
        raise
    except DatabaseOperationError as exc:
        raise _database_unavailable(exc) from exc
    except RecordNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio holding was not found.",
        ) from exc

    try:
        price = service.get_latest_price(holding.symbol)
    except Exception:
        logger.exception("Price service failed for portfolio symbol %s", holding.symbol)
        price = None
    return _holding_response(holding, price)


@router.delete(
    "/portfolio/holdings/{holding_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a portfolio holding",
    description="Remove a holding owned by the authenticated user.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "The holding does not exist."}
    },
)
def delete_holding(
    holding_id: Annotated[UUID, Path(description="ID returned when the holding was added")],
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> None:
    """Delete an existing holding."""
    try:
        database_holding = _owned_database_holding(
            db, holding_id, current_user.user_id
        )
        if database_holding is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio holding was not found.",
            )
        delete_database_holding(db, database_holding.holding_id)
    except HTTPException:
        raise
    except DatabaseOperationError as exc:
        raise _database_unavailable(exc) from exc
    except RecordNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio holding was not found.",
        ) from exc
