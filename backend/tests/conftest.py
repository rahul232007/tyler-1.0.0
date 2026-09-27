"""Shared pytest fixtures for JARVIS test suite."""
import asyncio
from typing import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.security import create_access_token, hash_password
from app.db.session import get_db_session
from app.main import app
from app.models import User
from app.tools.tools import register_stateless_tools

settings = get_settings()

# Create a test-isolated engine using NullPool to prevent asyncpg cross-loop connection errors
test_engine = create_async_engine(settings.database_url, poolclass=NullPool)
TestAsyncSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session", autouse=True)
def setup_test_tools():
    """Register stateless tools before running test suite."""
    register_stateless_tools()


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a clean isolated database session per test."""
    async with TestAsyncSessionLocal() as session:
        yield session


@pytest_asyncio.fixture(autouse=True)
def override_db_dependency(db_session):
    """Override FastAPI get_db_session dependency with the test session."""
    async def _get_test_db():
        yield db_session

    app.dependency_overrides[get_db_session] = _get_test_db
    yield
    app.dependency_overrides.pop(get_db_session, None)


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a temporary test user."""
    unique_email = f"test_{uuid4().hex[:8]}@jarvis.ai"
    user = User(
        email=unique_email,
        display_name="Test User",
        password_hash=hash_password("TestSecret123!"),
        preferences={"language": "english", "response_style": "concise"},
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def auth_headers(test_user: User) -> dict[str, str]:
    """Provide Bearer auth headers for test_user."""
    token = create_access_token(test_user.id, test_user.token_version)
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Provide an async test client connected to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
