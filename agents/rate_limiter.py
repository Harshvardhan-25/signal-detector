"""
Thread-safe token-bucket rate limiter.

Used to keep concurrent LLM calls (fired from LangGraph's .batch() running
several customer pipelines at once) under Gemini's requests-per-minute
quota. Without this, BATCH_SIZE > 1 combined with a low free-tier quota
(15 req/min for flash-lite) causes 429 RESOURCE_EXHAUSTED errors, since
several threads can call the API within the same second.

Every thread calls .acquire() right before hitting the API; if the quota
for the current rolling window is used up, that thread blocks (sleeps)
until a slot frees, rather than firing anyway and getting rejected.
"""
import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls = deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0])
            time.sleep(max(wait, 0.05))
