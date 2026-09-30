"""HTTP layer (FastAPI) + the in-process job manager."""

__all__ = ["app", "MANAGER"]


def __getattr__(name: str):  # lazy so importing src.api never needs fastapi
    if name == "app":
        from .server import app

        return app
    if name == "MANAGER":
        from .jobs import MANAGER

        return MANAGER
    raise AttributeError(name)
