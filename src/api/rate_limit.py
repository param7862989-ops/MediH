"""Minimal in-process rate limiting for abuse-prone endpoints (login, Gemini narratives).

Counts requests per client address in a sliding window. State lives in this process, which matches
the single-process Waitress deployment (wsgi.py); several processes would each keep their own counts.
Behind a reverse proxy set TRUST_PROXY_HOPS so the client address is the real one, not the proxy's.
"""

import threading
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Callable, Deque, Dict, Optional, Tuple

from flask import current_app, request

from src.api.schemas import error_response

_hits: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def _allow(bucket: str, key: str, limit: int, window_seconds: float) -> Tuple[bool, int]:
    now = time.monotonic()
    with _lock:
        hits = _hits[(bucket, key)]
        while hits and now - hits[0] >= window_seconds:
            hits.popleft()
        if len(hits) >= limit:
            return False, int(window_seconds - (now - hits[0])) + 1
        hits.append(now)
        return True, 0


def reset_rate_limits() -> None:
    with _lock:
        _hits.clear()


def rate_limited(bucket: str, limit: int, window_seconds: float, applies: Optional[Callable[[], bool]] = None):
    """Allows `limit` requests per client address per `window_seconds`, then answers 429.

    `applies` (optional) decides per request whether the limit counts at all.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if current_app.config.get("RATE_LIMITS_ENABLED", True) and (applies is None or applies()):
                allowed, retry_after = _allow(bucket, request.remote_addr or "unknown", limit, window_seconds)
                if not allowed:
                    response, status = error_response(
                        message=f"Too many requests. Try again in {retry_after} seconds.",
                        code="RATE_LIMITED", status_code=429)
                    response.headers["Retry-After"] = str(retry_after)
                    return response, status
            return fn(*args, **kwargs)
        return wrapper
    return decorator
