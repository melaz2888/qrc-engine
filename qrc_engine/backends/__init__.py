"""Quantum backend implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from qrc_engine.backends.base import BaseBackend

__all__ = ["BaseBackend", "QiskitBackend", "PercevalBackend", "DynamiqsBackend"]

_BACKEND_MODULES = {
    "QiskitBackend": "qrc_engine.backends.qiskit_backend",
    "PercevalBackend": "qrc_engine.backends.perceval_backend",
    "DynamiqsBackend": "qrc_engine.backends.dynamiqs_backend",
}


def __getattr__(name: str) -> Any:
    """Lazily import optional backends only when requested."""

    if name in _BACKEND_MODULES:
        module = import_module(_BACKEND_MODULES[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
