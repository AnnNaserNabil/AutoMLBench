### utils.py
import time


def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


def log_message(message):
    print(f"[AutoMLBench]: {message}")
