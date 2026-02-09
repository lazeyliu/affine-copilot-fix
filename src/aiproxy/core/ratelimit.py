"""Simple in-memory rate limiter."""
import threading
import time


class RateLimiter:
    """Fixed-window rate limiter for low-traffic proxies."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buckets = {}

    def allow(self, key, limit, window_seconds):
        """Return (allowed, remaining, reset_seconds) for the given key."""
        if not key or not limit or not window_seconds:
            return True, None, None

        now = time.time()
        bucket = int(now // window_seconds)
        reset_at = (bucket + 1) * window_seconds

        with self._lock:
            current_bucket, count = self._buckets.get(key, (bucket, 0))
            if current_bucket != bucket:
                current_bucket, count = bucket, 0
            count += 1
            self._buckets[key] = (current_bucket, count)

        remaining = max(0, limit - count)
        allowed = count <= limit
        reset_seconds = max(0, int(reset_at - now))
        return allowed, remaining, reset_seconds
