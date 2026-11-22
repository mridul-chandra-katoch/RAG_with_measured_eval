import json
import time
import math
import random
from pathlib import Path
from ragkit.claim_check import check_claims

def _percentile(data, p):
    if not data:
        return 0.0
    data.sort()
    idx = p * (len(data) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return data[lo]
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)

def _bootstrap_ci(values, n_resamples=1000, seed=42):
    """Bootstrap CI for a list of values (cluster by document later)."""
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n-1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return [_percentile(means, 0.025), _percentile(means, 0.975)]

def run_full_evaluation(pipeline, questions, config_name="default"):
    """Runs the full metric suite and saves a run record."""
    
    latencies = []
    faithfulness_scores = []
    citation_precisions = []
    
    tp, fp, tn, fn = 0, 0, 0, 0
    answerable_count = 0
    unanswerable_count = 0
    
    per_question_results = []
    
    for q in questions:
        start_t = time.time()
        ans = pipeline.ask(q["question"])
        latency = (time.time() - start_t) * 1000  # ms
        latencies.append(latency)
        
        is_answerable = q.get("answerable", True)
        if is_answerable:
            answerable_count += 1
        else:
            unanswerable_count += 1
            
        # Abstention metrics
        if not ans.abstained and is_answerable:
            tp += 1
        elif not ans.abstained and not is_answerable:
            fp += 1
        elif ans.abstained and not is_answerable:
            tn += 1
        elif ans.abstained and is_answerable:
            fn += 1
            
        # Faithfulness / Claim Check
        claim_res = check_claims(ans.answer, ans.citations, pipeline.retriever.tfidf.chunks)
        is_faithful = 1.0 if claim_res["overall_label"] == "cited" else 0.0
        faithfulness_scores.append(is_faithful)
        
        # Citation precision (if answered, are the citations relevant?)
        if not ans.abstained and ans.citations:
            # Check if any citation contains the gold span
            gold_span = q.get("gold_span")
            if gold_span:
                gold_norm = " ".join(gold_span.lower().split())
                has_gold = any(gold_norm in " ".join(c.quote.lower().split()) for c in ans.citations)
                citation_precisions.append(1.0 if has_gold else 0.0)
            else:
                citation_precisions.append(1.0) # No gold span to check against
        else:
            citation_precisions.append(0.0)
            
        per_question_results.append({
            "id": q["id"],
            "question": q["question"],
            "abstained": ans.abstained,
            "answer": ans.answer,
            "faithful": is_faithful,
            "latency_ms": latency
        })
        
    # Calculate aggregate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_refusal_rate = fn / answerable_count if answerable_count > 0 else 0.0
    false_accept_rate = fp / unanswerable_count if unanswerable_count > 0 else 0.0
    
    metrics = {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
        "citation_precision": sum(citation_precisions) / len(citation_precisions) if citation_precisions else 0.0,
        "abstention_precision": precision,
        "abstention_recall": recall,
        "false_refusal_rate": false_refusal_rate,
        "false_accept_rate": false_accept_rate,
        "p50_latency_ms": _percentile(latencies[:], 0.50),
        "p95_latency_ms": _percentile(latencies[:], 0.95),
        "faithfulness_ci": _bootstrap_ci(faithfulness_scores),
    }
    
    run_record = {
        "run_id": f"run_{int(time.time())}",
        "config_name": config_name,
        "metrics": metrics,
        "per_question": per_question_results
    }
    
    # Save artifacts
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{run_record['run_id']}.json").write_text(json.dumps(run_record, indent=2))
    
    return run_record, metrics