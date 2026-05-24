import ssl

from urllib.parse import quote_plus

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core import config

password = quote_plus(config.POSTGRES_PASSWORD)

# PG_URL = f"postgresql+asyncpg://{config.POSTGRES_USER}:{password}@{config.POSTGRES_HOST}:5432/{config.POSTGRES_DB}"
PG_URL = (
    f"postgresql+asyncpg://{config.POSTGRES_USER}:"
    f"{password}@{config.POSTGRES_HOST}:5432/{config.POSTGRES_DB}"
)
# ssl_context = ssl.create_default_context(
#     cafile="/opt/homebrew/etc/ca-certificates/cert.pem"
# )
ssl_context = ssl.create_default_context()

engine = create_async_engine(
    PG_URL,
    future=True,
    echo=False,
    pool_size=20,  # increase from 5
    max_overflow=20,  # allow bursts
    pool_timeout=60,  # wait up to 60s for a connection
    pool_recycle=600,  # recycle every 10 mins
    # connect_args={"ssl": ssl_context},
)

SessionFactory = async_sessionmaker(
    engine, autoflush=False, expire_on_commit=False, class_=AsyncSession
)


async def get_db():
    db = SessionFactory()
    try:
        yield db
    finally:
        await db.close()
