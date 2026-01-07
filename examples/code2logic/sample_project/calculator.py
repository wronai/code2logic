from __future__ import annotations


class Calculator:
    """A tiny calculator used for Code2Logic examples."""

    def add(self, a: float, b: float) -> float:
        """Return the sum of two numbers."""
        return a + b

    def divide(self, a: float, b: float) -> float:
        """Divide a by b.

        Raises:
            ZeroDivisionError: When b is 0.
        """
        if b == 0:
            raise ZeroDivisionError("b must not be 0")
        return a / b


def factorial(n: int) -> int:
    """Compute n! for n >= 0."""
    if n < 0:
        raise ValueError("n must be >= 0")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
