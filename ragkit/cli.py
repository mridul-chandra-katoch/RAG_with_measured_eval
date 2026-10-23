import argparse
import json
from pathlib import Path

def build_pipeline(config_path):
    from ragkit.config import Config
    from ragkit.chunk import chunk_documents
    from ragkit.index import TFIDFIndex, BM25Index
    from ragkit.retrieve import HybridRetriever
    from ragkit.generate import RAGPipeline
    cfg = Config.load(config_path)
    chunks = chunk_documents(Path(cfg.source_folder), cfg.max_tokens, cfg.overlap_tokens)
    tfidf = TFIDFIndex()
    tfidf.build(chunks)
    bm25 = BM25Index()
    bm25.build(chunks)
    retriever = HybridRetriever(tfidf, bm25, cfg.top_k)
    return RAGPipeline(retriever, cfg.abstain_threshold)

def main():
    parser = argparse.ArgumentParser(prog="ragkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask_p = sub.add_parser("ask")
    ask_p.add_argument("question")
    ask_p.add_argument("--config", default="configs/demo.json")

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--config", default="configs/demo.json")
    eval_p.add_argument("--questions", default="evals/fixture_questions.json")

    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--config", default="configs/demo.json")
    serve_p.add_argument("--port", type=int, default=8000)

    redteam_p = sub.add_parser("redteam")
    redteam_p.add_argument("--config", default="configs/demo.json")

    agent_p = sub.add_parser("agent")
    agent_p.add_argument("question")
    agent_p.add_argument("--config", default="configs/demo.json")

    args = parser.parse_args()

    if args.cmd == "ask":
        p = build_pipeline(args.config)
        ans = p.ask(args.question)
        print("Answer: " + ans.answer)
        print("Abstained: " + str(ans.abstained))
        print("Confidence: " + format(ans.confidence, ".3f"))
        print("Citations:")
        for c in ans.citations:
            print("  - " + c.file + " p" + str(c.page) + ": " + c.quote[:100] + "...")
    elif args.cmd == "eval":
        from ragkit.evals import run_eval
        from ragkit.schemas import Question
        p = build_pipeline(args.config)
        raw = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        qs = [Question(**q) for q in raw]
        result = run_eval(qs, p.retriever, p)
        for k, v in result.items():
            print(k + ": " + str(v))
    elif args.cmd == "serve":
        from ragkit.serve import serve
        p = build_pipeline(args.config)
        serve(p, args.port)
    elif args.cmd == "redteam":
        from ragkit.redteam import run_redteam
        p = build_pipeline(args.config)
        results = run_redteam(p)
        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        print("RedTeam Results: " + str(passed) + "/" + str(total) + " passed")
        print("Attack Success Rate: " + format((total - passed) / total * 100, ".1f") + "%\n")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print("  " + r["attack"] + ": " + status)
            if not r["passed"] and "error" in r:
                print("    Error: " + r["error"])
    elif args.cmd == "agent":
        from ragkit.agent import run_agent
        p = build_pipeline(args.config)
        ans, tool = run_agent(args.question, p)
        print("[" + tool + "] " + ans.answer)

if __name__ == "__main__":
    main()