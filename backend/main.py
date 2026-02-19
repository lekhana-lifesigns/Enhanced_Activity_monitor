"""
Enhanced Activity Monitor - Enterprise API
==========================================

FastAPI-based REST API for clinical activity monitoring.

Features:
- JWT authentication with role-based access control
- RESTful endpoints for events, alerts, devices, patients
- WebSocket for real-time streaming
- OpenAPI documentation
- Health checks and monitoring
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.core.config import settings
from backend.core.database import init_db, close_db
from backend.api.v1.router import api_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    log.info("Starting Enhanced Activity Monitor API...")
    log.info(f"Environment: {settings.ENVIRONMENT}")
    log.info(f"Debug: {settings.DEBUG}")

    # Initialize database
    await init_db()
    log.info("Database initialized")

    # TODO: Initialize MQTT consumer
    # TODO: Initialize Redis connection

    log.info("API ready to serve requests")

    yield

    # Shutdown
    log.info("Shutting down API...")
    await close_db()
    log.info("API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url=settings.DOCS_URL if settings.DEBUG else None,
    redoc_url=settings.REDOC_URL if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    log.exception(f"Unexpected error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# Health check endpoint (public)
@app.get("/health", tags=["Health"])
async def health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with component status."""
    from sqlalchemy import text
    from backend.core.database import AsyncSessionLocal
    import psutil
    import time

    checks = {
        "api": {"status": "healthy", "latency_ms": 0},
        "database": {"status": "unknown", "latency_ms": 0},
        "system": {},
    }

    # Database check
    start = time.time()
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        checks["database"]["status"] = "healthy"
        checks["database"]["latency_ms"] = round((time.time() - start) * 1000, 2)
    except Exception as e:
        checks["database"]["status"] = "unhealthy"
        checks["database"]["error"] = str(e)

    # System metrics
    checks["system"] = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
    }

    overall = "healthy" if all(
        c.get("status") == "healthy" for c in [checks["api"], checks["database"]]
    ) else "degraded"

    return {
        "status": overall,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
    }


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    import psutil
    from fastapi.responses import PlainTextResponse

    # Collect metrics
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    metrics = []

    # System metrics
    metrics.append(f"# HELP eam_cpu_usage_percent CPU usage percentage")
    metrics.append(f"# TYPE eam_cpu_usage_percent gauge")
    metrics.append(f'eam_cpu_usage_percent{{instance="api"}} {cpu}')

    metrics.append(f"# HELP eam_memory_usage_percent Memory usage percentage")
    metrics.append(f"# TYPE eam_memory_usage_percent gauge")
    metrics.append(f'eam_memory_usage_percent{{instance="api"}} {mem.percent}')

    metrics.append(f"# HELP eam_memory_used_bytes Memory used in bytes")
    metrics.append(f"# TYPE eam_memory_used_bytes gauge")
    metrics.append(f'eam_memory_used_bytes{{instance="api"}} {mem.used}')

    metrics.append(f"# HELP eam_disk_usage_percent Disk usage percentage")
    metrics.append(f"# TYPE eam_disk_usage_percent gauge")
    metrics.append(f'eam_disk_usage_percent{{instance="api"}} {disk.percent}')

    # Application info
    metrics.append(f"# HELP eam_info Application information")
    metrics.append(f"# TYPE eam_info gauge")
    metrics.append(f'eam_info{{version="{settings.APP_VERSION}",environment="{settings.ENVIRONMENT}"}} 1')

    return PlainTextResponse("\n".join(metrics), media_type="text/plain")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": settings.DOCS_URL,
        "api": settings.API_V1_PREFIX,
    }


# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS if not settings.RELOAD else 1,
    )
