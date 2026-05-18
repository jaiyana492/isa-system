import logging
import re
import ssl as ssl_module

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm import DeclarativeBase
from config.settings import settings

logger = logging.getLogger(__name__)

_ssl_ctx = ssl_module.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl_module.CERT_NONE


def _normalize_db_url(url: str) -> str:
    # Strip invisible whitespace/newlines that can come from copy-paste
    url = url.strip()

    # Replace any scheme variant with the one asyncpg requires
    url = re.sub(
        r"^postgres(?:ql)?(\+[^:]+)?://",
        "postgresql+asyncpg://",
        url,
    )
    return url


_raw_url = settings.DATABASE_URL
_db_url = _normalize_db_url(_raw_url)

# Log scheme only — never log passwords
logger.info("DATABASE | raw scheme=%s  normalized scheme=%s",
            _raw_url.split("://")[0] if "://" in _raw_url else "MISSING",
            _db_url.split("://")[0] if "://" in _db_url else "MISSING")

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
