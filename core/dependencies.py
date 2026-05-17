from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from config.database import AsyncSessionLocal


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a transactional database session per request.
    Automatically commits on success.
    Automatically rolls back on exception.
    Automatically closes after request completes.
    """
    async with AsyncSessionLocal() as session:
        async with session.begin():
            yield session