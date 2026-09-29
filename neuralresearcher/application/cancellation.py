import asyncio
from typing import Callable

class CancellationToken:
    def __init__(self):
        self._is_cancelled = False
        self._callbacks: list[Callable[[], None]] = []

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def cancel(self):
        if not self._is_cancelled:
            self._is_cancelled = True
            for callback in self._callbacks:
                try:
                    callback()
                except Exception:
                    pass

    def register(self, callback: Callable[[], None]):
        if self._is_cancelled:
            callback()
        else:
            self._callbacks.append(callback)

    def check(self):
        """Raises CancelledError if cancellation was requested."""
        if self._is_cancelled:
            raise asyncio.CancelledError("Operation was cancelled.")
