import argparse
import json
import sys
import hashlib
from pathlib import Path

from ragkit.db import MetadataDB

def build_pipeline(config_path):
    from ragkit.config import CorpusConfig
    from ragkit.index import TFIDFIndex, BM25Index
    from ragkit.retrieve import HybridRetriever
    from ragkit.generate import RAGPipeline
    
    cfg = CorpusConfig.load(config_path)
    
    # Load chunks from DB (much faster!)
    db = MetadataDB()
    chunks = db.get_all_chunks()
    
    # If empty, fallback to parsing (but warn user to run ingest)
    if not chunks:
        from ragkit.chunk import chunk_documents
        source_path = cfg.sources[0]["path"]
        chunks = chunk_documents(Path(source_path), cfg.max_tokens, cfg.overlap_tokens, cfg.chunking_strategy)
        print("WARNING: DB is empty. Falling back to live parsing. Run 'ragkit ingest <folder>' to save to DB.")
    
    tfidf = TFIDFIndex()
    tfidf.build(chunks)
    bm25 = BM25Index()
    bm25.build(chunks)
    
    retriever = HybridRetriever(tfidf, bm25, cfg.retrieval_top_k)
    return RAGPipeline(retriever, cfg.abstain_threshold, cfg.abstain_mode)

def main():
    parser = argparse.ArgumentParser(prog="ragkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1. Ingest
    ingest_p = sub.add_parser("ingest")
    ingest_p.add_argument("folder")
    ingest_p.add_argument("--config", default="configs/corpus.yaml")

    # 2. Ask
    ask_p = sub.add_parser("ask")
    ask_p.add_argument("question")
    ask_p.add_argument("--config", default="configs/corpus.yaml")

    # 3. Serve
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--config", default="configs/corpus.yaml")
    serve_p.add_argument("--port", type=int, default=8000)

    # 4. Eval
    eval_p = sub.add_parser("eval")
    eval_sub = eval_p.add_subparsers(dest="eval_cmd", required=True)
    eval_run = eval_sub.add_parser("run")
    eval_run.add_argument("--config", default="configs/corpus.yaml")
    eval_run.add_argument("--questions", default="evals/fixture_questions.json")
    eval_run.add_argument("--split", default="dev", choices=["dev", "test"])

    # 5. Redteam
    redteam_p = sub.add_parser("redteam")
    redteam_p.add_argument("action", choices=["run"], default="run", nargs="?")
    redteam_p.add_argument("--no-defense", action="store_true")
    redteam_p.add_argument("--config", default="configs/corpus.yaml")

    # 6. Train
    train_p = sub.add_parser("train")
    train_p.add_argument("model", choices=["abstain", "reranker"])
    train_p.add_argument("--config", default="configs/corpus.yaml")

    # 7. Report
    report_p = sub.add_parser("report")
    report_p.add_argument("run_id")

    # 8. Agent (Legacy/Internal)
    agent_p = sub.add_parser("agent")
    agent_p.add_argument("question")
    agent_p.add_argument("--config", default="configs/corpus.yaml")

    args = parser.parse_args()

    if args.cmd == "ingest":
        from ragkit.config import CorpusConfig
        from ragkit.chunk import chunk_documents
        
        cfg = CorpusConfig.load(args.config)
        folder = Path(args.folder)
        db = MetadataDB()
        
        print(f"Parsing folder {folder}...")
        # Parse the folder ONCE
        all_chunks = chunk_documents(folder, cfg.max_tokens, cfg.overlap_tokens, cfg.chunking_strategy)
        
        # Group chunks by file
        chunks_by_file = {}
        for c in all_chunks:
            chunks_by_file.setdefault(c.source["file"], []).append(c)
            
        files_processed = 0
        files_skipped = 0
        
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf", ".html", ".htm", ".docx"}:
                continue
                
            file_str = str(path.relative_to(folder))
            file_bytes = path.read_bytes()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            doc_id = file_str.replace("/", "_").replace("\\", "_").replace(".", "_")
            
            existing_hash = db.get_file_hash(doc_id)
            if existing_hash == file_hash:
                files_skipped += 1
                continue
                
            print(f"Processing {path.name}...")
            doc_chunks = chunks_by_file.get(file_str, [])
            db.add_document_with_chunks(doc_id, file_str, file_hash, doc_chunks)
            files_processed += 1
            
        print(f"Ingest complete: {files_processed} files processed, {files_skipped} skipped (unchanged).")
        
    elif args.cmd == "ask":
        p = build_pipeline(args.config)
        ans = p.ask(args.question)
        print("Answer: " + ans.answer)
        print("Abstained: " + str(ans.abstained))
        print("Confidence: " + format(ans.confidence, ".3f"))
        print("Citations:")
        for c in ans.citations:
            print("  - " + c.file + " p" + str(c.page) + ": " + c.quote[:100] + "...")
            
    elif args.cmd == "serve":
        from ragkit.serve import serve
        p = build_pipeline(args.config)
        serve(p, args.port)
        
    elif args.cmd == "eval":
        if args.eval_cmd == "run":
            from ragkit.evals import run_eval
            from ragkit.schemas import Question
            p = build_pipeline(args.config)
            raw = json.loads(Path(args.questions).read_text(encoding="utf-8"))
            qs = [Question(**q) for q in raw]
            result = run_eval(qs, p.retriever, p)
            for k, v in result.items():
                print(k + ": " + str(v))
        else:
            print(f"Eval subcommand '{args.eval_cmd}' not fully implemented yet.")
            
    elif args.cmd == "redteam":
        from ragkit.redteam import run_redteam
        p = build_pipeline(args.config)
        results = run_redteam(p, defense_enabled=not args.no_defense)
        
        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        print("RedTeam Results: " + str(passed) + "/" + str(total) + " passed")
        print("Attack Success Rate: " + format((total - passed) / total * 100, ".1f") + "%\n")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print("  " + r["attack"] + ": " + status)
            
    elif args.cmd == "train":
        if args.model == "abstain":
            from ragkit.learned.train_abstention import main as train_abstain
            sys.argv = ["train_abstention", "--max-articles", "150", "--seed", "42"]
            train_abstain()
        elif args.model == "reranker":
            from ragkit.learned.train_reranker import main as train_reranker
            sys.argv = ["train_reranker", "--max-articles", "150", "--seed", "42"]
            train_reranker()

    elif args.cmd == "report":
        print(f"Fetching report for run_id: {args.run_id} (Not implemented yet)")
        
    elif args.cmd == "agent":
        from ragkit.agent import run_agent
        p = build_pipeline(args.config)
        ans, tool = run_agent(args.question, p)
        print("[" + tool + "] " + ans.answer)

if __name__ == "__main__":
    main()