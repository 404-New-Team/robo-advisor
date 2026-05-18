import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.main import app
from app.database import get_db, Base
from app.models import user as _user_models  # noqa: F401 — register models


@pytest.fixture
def auth_client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    saved = app.dependency_overrides.get(get_db)

    def _db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _db
    with TestClient(app) as c:
        yield c
    if saved is not None:
        app.dependency_overrides[get_db] = saved
    else:
        app.dependency_overrides.pop(get_db, None)


_REG = {"email": "test@example.com", "username": "tester", "password": "secret123"}


def test_register_success(auth_client):
    r = auth_client.post("/auth/register", json=_REG)
    assert r.status_code == 201
    data = r.json()
    assert data["email"] == _REG["email"]
    assert data["username"] == _REG["username"]
    assert "id" in data


def test_register_duplicate_email(auth_client):
    auth_client.post("/auth/register", json=_REG)
    r = auth_client.post("/auth/register", json=_REG)
    assert r.status_code == 409


def test_register_invalid_email(auth_client):
    r = auth_client.post("/auth/register", json={**_REG, "email": "not-an-email"})
    assert r.status_code == 422


def test_register_missing_field(auth_client):
    r = auth_client.post("/auth/register", json={"email": _REG["email"], "username": _REG["username"]})
    assert r.status_code == 422


def test_login_success(auth_client):
    auth_client.post("/auth/register", json=_REG)
    r = auth_client.post("/auth/login", json={"email": _REG["email"], "password": _REG["password"]})
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(auth_client):
    auth_client.post("/auth/register", json=_REG)
    r = auth_client.post("/auth/login", json={"email": _REG["email"], "password": "wrongpass"})
    assert r.status_code == 401


def test_login_unknown_email(auth_client):
    r = auth_client.post("/auth/login", json={"email": "nobody@example.com", "password": "x"})
    assert r.status_code == 401


def test_me_success(auth_client):
    auth_client.post("/auth/register", json=_REG)
    token = auth_client.post("/auth/login", json={"email": _REG["email"], "password": _REG["password"]}).json()["access_token"]
    r = auth_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["email"] == _REG["email"]


def test_me_no_token(auth_client):
    r = auth_client.get("/auth/me")
    assert r.status_code == 401


def test_me_invalid_token(auth_client):
    r = auth_client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
    assert r.status_code == 401
