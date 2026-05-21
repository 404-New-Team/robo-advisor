import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.main import app
from app.database import get_db, Base
from app.auth import get_current_user
from app.models.user import User
from app.models import user as _user_models  # noqa: F401 — register models


@pytest.fixture
def users_client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    db = Session()
    db.add(User(id=1, email="u@example.com", username="tuser", password_hash="x", created_at=datetime.utcnow()))
    db.commit()
    db.close()

    saved_db = app.dependency_overrides.get(get_db)
    saved_auth = app.dependency_overrides.get(get_current_user)

    mock_user = User(id=1, email="u@example.com", username="tuser", password_hash="x", created_at=datetime.utcnow())

    def _db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_current_user] = lambda: mock_user

    with TestClient(app) as c:
        yield c

    if saved_db is not None:
        app.dependency_overrides[get_db] = saved_db
    else:
        app.dependency_overrides.pop(get_db, None)

    if saved_auth is not None:
        app.dependency_overrides[get_current_user] = saved_auth
    else:
        app.dependency_overrides.pop(get_current_user, None)


def test_get_tickers_empty(users_client):
    r = users_client.get("/users/tickers")
    assert r.status_code == 200
    assert r.json()["tickers"] == []


def test_add_ticker_success(users_client):
    r = users_client.post("/users/tickers", json={"ticker": "AAPL"})
    assert r.status_code == 201
    assert "AAPL" in r.json()["tickers"]


def test_add_ticker_stored_uppercase(users_client):
    r = users_client.post("/users/tickers", json={"ticker": "tsla"})
    assert r.status_code == 201
    assert "TSLA" in r.json()["tickers"]


def test_add_multiple_tickers(users_client):
    users_client.post("/users/tickers", json={"ticker": "SPY"})
    r = users_client.post("/users/tickers", json={"ticker": "QQQ"})
    tickers = r.json()["tickers"]
    assert "SPY" in tickers
    assert "QQQ" in tickers


def test_add_duplicate_ticker(users_client):
    users_client.post("/users/tickers", json={"ticker": "MSFT"})
    r = users_client.post("/users/tickers", json={"ticker": "msft"})
    assert r.status_code == 409


def test_delete_ticker_success(users_client):
    users_client.post("/users/tickers", json={"ticker": "NVDA"})
    r = users_client.delete("/users/tickers/NVDA")
    assert r.status_code == 200
    assert "NVDA" not in r.json()["tickers"]


def test_delete_ticker_case_insensitive(users_client):
    users_client.post("/users/tickers", json={"ticker": "GOOG"})
    r = users_client.delete("/users/tickers/goog")
    assert r.status_code == 200
    assert "GOOG" not in r.json()["tickers"]


def test_delete_nonexistent_ticker(users_client):
    r = users_client.delete("/users/tickers/FAKE")
    assert r.status_code == 404
