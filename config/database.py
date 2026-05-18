import ssl as ssl_module

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm import DeclarativeBase
from config.settings import settings

# Supabase uses SSL but Windows doesn't have their CA in the default trust store.
# We require SSL (traffic is encrypted) but skip chain verification for dev.
_ssl_ctx = ssl_module.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl_module.CERT_NONE

# Normalize DATABASE_URL — Supabase provides postgres:// or postgresql://
# but asyncpg requires the postgresql+asyncpg:// scheme.
def _normalize_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://"):]
    return url

engine = create_async_engine(
    _normalize_db_url(settings.DATABASE_URL),
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={
        "statement_cache_size": 0,  # required for Supabase PgBouncer transaction mode
        "ssl": _ssl_ctx,
    },
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


class Base(DeclarativeBase):
    pass


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_engine():
    return engine
