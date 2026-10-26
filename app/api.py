from __future__ import annotations

import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from mlforge.config import Settings
from mlforge.errors import ForgeError
from mlforge.schemas import ExperimentConfig, PredictionRequest, PredictionResponse
from mlforge.service import Forge
from mlforge.storage import now, uid

logger = logging.getLogger("mlforge.api")


class BodyTooLarge(Exception):
    pass


class BodyLimit:
    def __init__(self, app, maximum: int):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        scope.setdefault("state", {})["request_started"] = time.perf_counter()
        used = 0

        async def bounded_receive():
            nonlocal used
            message = await receive()
            used += len(message.get("body", b""))
            if used > self.maximum:
                raise BodyTooLarge
            return message

        try:
            length = dict(scope.get("headers", [])).get(b"content-length")
            if length is not None and (not length.isdigit() or int(length) > self.maximum):
                raise BodyTooLarge
            chunks = []
            while True:
                message = await bounded_receive()
                if message["type"] == "http.disconnect":
                    return
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            body, delivered = b"".join(chunks), False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay, send)
        except BodyTooLarge:
            await JSONResponse(
                {"error": "too_large", "message": "Request exceeds the 1 MiB limit"},
                status_code=413,
            )(scope, receive, send)


class Registration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    name: str = Field(min_length=1, max_length=100)


def create_app(settings: Settings | None = None) -> FastAPI:
    forge = Forge(settings)
    app = FastAPI(
        title="MLForge local prediction API",
        version="1.0.0",
        description="CPU-only local tabular model registry and predictions. No arbitrary artifact uploads.",
    )
    app.state.forge = forge
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
    )
    app.add_middleware(BodyLimit, maximum=forge.settings.max_request_bytes)

    @app.exception_handler(ForgeError)
    async def forge_error(request: Request, exc: ForgeError):
        return JSONResponse({"error": exc.code, "message": str(exc)}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if request.url.path.startswith("/predict/"):
            forge.store.save(
                "prediction",
                {
                    "id": uid(),
                    "created_at": now(),
                    "model_id": request.url.path.rsplit("/", 1)[-1][:64],
                    "rows": 0,
                    "status": "failed",
                    "duration_ms": (time.perf_counter() - request.state.request_started) * 1000,
                    "error": "Invalid prediction request format",
                },
            )
            forge.store.retain_logs(forge.settings.max_logs, forge.settings.retention_days)
        errors = [
            {"location": list(e["loc"]), "message": e["msg"], "type": e["type"]}
            for e in exc.errors()[:20]
        ]
        return JSONResponse(
            {"error": "invalid_request", "message": "Request validation failed", "details": errors},
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def server_error(request: Request, exc: Exception):
        event = uid()
        logger.error(
            json.dumps(
                {"event": "request_failed", "event_id": event, "exception_type": type(exc).__name__}
            )
        )
        return JSONResponse(
            {
                "error": "internal_error",
                "message": "Request could not be completed",
                "event_id": event,
            },
            status_code=500,
        )

    @app.get("/health")
    def health():
        forge.recover_jobs()
        return {
            "status": "ok",
            "version": "1.0.0",
            "training_concurrency": 1,
            "storage": "local SQLite",
            "cpu_only": True,
        }

    @app.get("/models")
    def models():
        return forge.store.list("model")

    @app.get("/models/{model_id}")
    def model(model_id: str):
        return forge.store.get("model", model_id)

    @app.post("/models", status_code=201)
    def register(body: Registration):
        return forge.register(body.experiment_id, body.name)

    @app.post("/predict/{model_id}", response_model=PredictionResponse)
    def predict(model_id: str, body: PredictionRequest):
        return forge.predict(model_id, body.rows)

    @app.get("/experiments")
    def experiments():
        forge.recover_jobs()
        return forge.store.list("experiment")

    @app.get("/experiments/{experiment_id}")
    def experiment(experiment_id: str):
        return forge.store.get("experiment", experiment_id)

    @app.post("/experiments/validate")
    def validate(body: ExperimentConfig):
        return forge.preflight(body)

    @app.post("/experiments", status_code=202)
    def start(body: ExperimentConfig):
        return forge.start(body)

    @app.post("/experiments/{experiment_id}/cancel")
    def cancel(experiment_id: str):
        forge.cancel(experiment_id)
        return {"status": "cancelled"}

    @app.get("/datasets")
    def datasets():
        return forge.store.list("dataset")

    @app.get("/monitoring")
    def monitoring():
        return {
            "retention": {
                "maximum_requests": forge.settings.max_logs,
                "days": forge.settings.retention_days,
            },
            "requests": forge.logs(),
        }

    return app
