import json
import time
from pathlib import Path

def log_trace(trace_id, query, latency_ms, cost, retrieved_chunk_ids, user_feedback=None):
    Path("results").mkdir(exist_ok=True)
    trace = {
        "trace_id": trace_id,
        "query": query,
        "latency_ms": latency_ms,
        "cost": cost,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "feedback": user_feedback,
        "timestamp": time.time()
    }
    with open("results/traces.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")

def get_latency_percentiles(traces_path="results/traces.jsonl"):
    latencies = []
    if not Path(traces_path).exists():
        return 0.0, 0.0
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                latencies.append(json.loads(line)["latency_ms"])
    
    if not latencies:
        return 0.0, 0.0
        
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    return p50, p95

def check_drift(old_hashes: dict, new_hashes: dict) -> bool:
    """Returns True if documents have changed significantly."""
    if set(old_hashes.keys()) != set(new_hashes.keys()):
        return True
    for k, v in old_hashes.items():
        if new_hashes[k] != v:
            return True
    return False