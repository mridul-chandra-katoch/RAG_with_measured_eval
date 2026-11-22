import argparse
import json
import sys
import hashlib
from pathlib import Path

from ragkit.db import MetadataDB

PARSER_VERSION = "1.0"

def build_pipeline(config_path):
    from ragkit.config import CorpusConfig
    from ragkit.index import TFIDFIndex, BM25Index
    from ragkit.retrieve import HybridRetriever
    from ragkit.generate import RAGPipeline
    
    cfg = CorpusConfig.load(config_path)
    db = MetadataDB()
    chunks = db.get_all_chunks()
    
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
    
    return RAGPipeline(
        retriever, 
        cfg.abstain_threshold, 
        cfg.abstain_mode, 
        cfg.provider_name, 
        cfg.reranker_enabled
    )

def main():
    parser = argparse.ArgumentParser(prog="ragkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_p = sub.add_parser("ingest")
    ingest_p.add_argument("folder")
    ingest_p.add_argument("--config", default="configs/corpus.yaml")

    ask_p = sub.add_parser("ask")
    ask_p.add_argument("question")
    ask_p.add_argument("--config", default="configs/corpus.yaml")

    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--config", default="configs/corpus.yaml")
    serve_p.add_argument("--port", type=int, default=8000)

    eval_p = sub.add_parser("eval")
    eval_sub = eval_p.add_subparsers(dest="eval_cmd", required=True)
    eval_run = eval_sub.add_parser("run")
    eval_run.add_argument("--config", default="configs/corpus.yaml")
    eval_run.add_argument("--questions", default="evals/questions.jsonl")
    eval_run.add_argument("--split", default="dev", choices=["dev", "test"])

    redteam_p = sub.add_parser("redteam")
    redteam_p.add_argument("action", choices=["run"], default="run", nargs="?")
    redteam_p.add_argument("--no-defense", action="store_true")
    redteam_p.add_argument("--config", default="configs/corpus.yaml")

    train_p = sub.add_parser("train")
    train_p.add_argument("model", choices=["abstain", "reranker"])
    train_p.add_argument("--config", default="configs/corpus.yaml")

    report_p = sub.add_parser("report")
    report_p.add_argument("run_id")

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
        
        chunk_config_str = f"{cfg.max_tokens}_{cfg.overlap_tokens}_{cfg.chunking_strategy}"
        chunk_config_hash = hashlib.sha256(chunk_config_str.encode()).hexdigest()
        
        print(f"Parsing folder {folder}...")
        all_chunks = chunk_documents(folder, cfg.max_tokens, cfg.overlap_tokens, cfg.chunking_strategy)
        
        chunks_by_file = {}
        for c in all_chunks:
            chunks_by_file.setdefault(c.source["file"], []).append(c)
            
        files_processed = 0
        files_skipped = 0
        current_files = set()
        
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf", ".html", ".htm", ".docx"}:
                continue
                
            file_str = str(path.relative_to(folder))
            current_files.add(file_str)
            
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            doc_id = file_str.replace("/", "_").replace("\\", "_").replace(".", "_")
            
            cache_key = hashlib.sha256(f"{file_hash}_{PARSER_VERSION}_{chunk_config_hash}".encode()).hexdigest()
            
            existing_key = db.get_cache_key(doc_id)
            if existing_key == cache_key:
                files_skipped += 1
                continue
                
            print(f"Processing {path.name}...")
            doc_chunks = chunks_by_file.get(file_str, [])
            db.add_document_with_chunks(doc_id, file_str, cache_key, doc_chunks)
            files_processed += 1
            
        current_doc_ids = {f.replace("/", "_").replace("\\", "_").replace(".", "_") for f in current_files}
        deleted_count = 0
        for existing_doc_id in db.get_all_doc_ids():
            if existing_doc_id not in current_doc_ids:
                db.delete_document(existing_doc_id)
                deleted_count += 1
                
        print(f"Ingest complete: {files_processed} files processed, {files_skipped} skipped, {deleted_count} deleted.")
        
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
            from ragkit.evals import run_benchmark
            p = build_pipeline(args.config)
            
            qs = []
            with open(args.questions, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        qs.append(json.loads(line))
                        
            result = run_benchmark(qs, p.retriever, k=5)
            print(f"--- Evaluation Results ({args.split}) ---")
            for k, v in result.items():
                print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
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