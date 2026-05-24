import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.router.rag import legacy_router as legacy_rag_router
from src.router.rag import router as rag_router
from src.router.user import router as auth_router

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG BMS API")
    yield
    logger.info("Shutting down RAG BMS API")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG BMS API",
        version="1.0.0",
        description="Production-ready RAG API using FastAPI",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict this before deploying publicly.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router)
    app.include_router(rag_router)
    app.include_router(legacy_rag_router)

    @app.get("/health", tags=["system"])
    async def health_check():
        return {"status": "healthy"}

    @app.get("/", tags=["system"])
    async def root():
        return {
            "message": "RAG BMS API is running",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
