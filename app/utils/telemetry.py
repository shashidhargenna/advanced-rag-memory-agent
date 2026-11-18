from time import perf_counter


def start_timer() -> float:
    """
    Start a performance timer.
    """
    return perf_counter()


def stop_timer(start_time: float) -> float:
    """
    Stop the timer and return elapsed time in milliseconds.
    """
    elapsed_time = perf_counter() - start_time

    return round(elapsed_time * 1000, 2)


def create_telemetry(
    retrieval_latency_ms: float = 0,
    rerank_latency_ms: float = 0,
    generation_latency_ms: float = 0,
    total_latency_ms: float = 0
) -> dict:
    """
    Create a structured telemetry response.
    """

    return {
        "retrieval_latency_ms": retrieval_latency_ms,
        "rerank_latency_ms": rerank_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "total_latency_ms": total_latency_ms
    }