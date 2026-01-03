"""Sample utilities for re-export testing."""

from typing import Any, Dict


def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process input data."""
    return {k: v for k, v in data.items() if v is not None}


def validate_input(data: Dict[str, Any]) -> bool:
    """Validate input data."""
    return bool(data) and all(isinstance(k, str) for k in data.keys())
