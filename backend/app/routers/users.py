from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models.user import User, UserTicker

router = APIRouter(prefix="/users", tags=["Users"])


class TickerItem(BaseModel):
    ticker: str


class TickerListResponse(BaseModel):
    tickers: list[str]


@router.get("/tickers", response_model=TickerListResponse, summary="내 ticker 목록 조회")
def get_tickers(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(UserTicker).filter(UserTicker.user_id == current_user.id).all()
    return TickerListResponse(tickers=[r.ticker for r in rows])


@router.post("/tickers", response_model=TickerListResponse, status_code=status.HTTP_201_CREATED, summary="ticker 추가")
def add_ticker(body: TickerItem, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ticker = body.ticker.upper().strip()
    exists = db.query(UserTicker).filter(UserTicker.user_id == current_user.id, UserTicker.ticker == ticker).first()
    if exists:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"{ticker}은(는) 이미 등록된 ticker입니다.")
    db.add(UserTicker(user_id=current_user.id, ticker=ticker))
    db.commit()
    rows = db.query(UserTicker).filter(UserTicker.user_id == current_user.id).all()
    return TickerListResponse(tickers=[r.ticker for r in rows])


@router.delete("/tickers/{ticker}", response_model=TickerListResponse, summary="ticker 삭제")
def delete_ticker(ticker: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ticker = ticker.upper().strip()
    row = db.query(UserTicker).filter(UserTicker.user_id == current_user.id, UserTicker.ticker == ticker).first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{ticker}을(를) 찾을 수 없습니다.")
    db.delete(row)
    db.commit()
    rows = db.query(UserTicker).filter(UserTicker.user_id == current_user.id).all()
    return TickerListResponse(tickers=[r.ticker for r in rows])
