from __future__ import annotations

from dataclasses import dataclass


@dataclass
class User:
    """Simple user model."""

    id: str
    name: str
    email: str
    active: bool = True
