import logging
import re
import ssl as _ssl_mod

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm import DeclarativeBase
from config.settings import settings

logger = logging.getLogger(__name__)


def _normalize_db_url(url: str) -> str:
    url = url.strip()
    url = re.sub(r"^postgres(?:ql)?(?:\+[^:]+)?://", "postgresql+asyncpg://", url)
    return url


_db_url = _normalize_db_url(settings.DATABASE_URL)
logger.info("DATABASE | scheme=%s", _db_url.split("://")[0])

_ssl_ctx = _ssl_mod.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = _ssl_mod.CERT_NONE

engine = create_async_engine(
    _db_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={
        "statement_cache_size": 0,
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
