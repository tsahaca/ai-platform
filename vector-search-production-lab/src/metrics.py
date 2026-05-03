import time
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    result = {"elapsed_ms": None}
    try:
        yield result
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
