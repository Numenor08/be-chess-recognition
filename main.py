from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analysis, health, recognize
from core.config import ALLOWED_ORIGINS
from core.lifecycle import on_shutdown, on_startup

app = FastAPI(title="Chess Modular API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.router.add_event_handler("startup", lambda: on_startup(app))
app.router.add_event_handler("shutdown", lambda: on_shutdown(app))

app.include_router(health.router)
app.include_router(recognize.router)
app.include_router(analysis.router)
