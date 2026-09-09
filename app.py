"""
FastAPI Backend Server for RAG Eval Suite Web Dashboard.
Serves static frontend (HTML, CSS, JS) and provides interactive RAG playground,
benchmark inspection, and golden dataset synthesis APIs.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure UTF-8 output encoding for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Root directory of the repository
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.generator import generate_answer
from components.retriever import build_retriever, retrieve_contexts
from evals.scripts.generate_goldens import generate_golden_dataset

load_dotenv()

app = FastAPI(
    title="RAG Eval Suite API",
    description="Backend API for RAG evaluation, live query testing, and benchmarking.",
    version="0.5.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for CSS and JS
static_dir = ROOT / "web" / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Frontend Root Route
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = ROOT / "web" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_file))


# ---------------------------------------------------------------------------
# API: Available Documents
# ---------------------------------------------------------------------------
@app.get("/api/documents")
async def list_documents():
    """Lists available source documents in the docs/ directory."""
    docs_dir = ROOT / "docs"
    if not docs_dir.exists():
        return []

    valid_exts = (".pdf", ".txt", ".md", ".json")
    documents = []
    for f in sorted(docs_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in valid_exts:
            documents.append({
                "name": f.name,
                "path": str(Path("docs") / f.name),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return documents


# ---------------------------------------------------------------------------
# API: Benchmark Results Viewer
# ---------------------------------------------------------------------------
@app.get("/api/results")
async def list_results():
    """Lists all saved evaluation runs in evals/results/ with summary stats."""
    results_dir = ROOT / "evals" / "results"
    if not results_dir.exists():
        return []

    runs = []
    for f in sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as rf:
                data = json.load(rf)

            run_type = "eval"
            if f.name.startswith("ops_run_"):
                run_type = "ops"
            elif "eval_type" in data.get("hyperparameters", {}):
                run_type = data["hyperparameters"]["eval_type"]

            runs.append({
                "filename": f.name,
                "type": run_type,
                "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
                "duration": data.get("runDuration", None),
                "cost": data.get("evaluationCost", None),
            })
        except Exception:
            continue

    return runs


@app.get("/api/results/{filename}")
async def get_result_detail(filename: str):
    """Returns the full JSON content of a specific evaluation run."""
    safe_name = Path(filename).name
    target_file = ROOT / "evals" / "results" / safe_name

    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"Run file '{safe_name}' not found.")

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read result file: {str(e)}")


# ---------------------------------------------------------------------------
# API: Golden Datasets
# ---------------------------------------------------------------------------
@app.get("/api/datasets")
async def list_datasets():
    """Lists available golden datasets in evals/datasets/."""
    datasets_dir = ROOT / "evals" / "datasets"
    if not datasets_dir.exists():
        return []

    datasets = []
    for f in sorted(datasets_dir.glob("*.json")):
        if f.name.endswith(".checkpoint.json"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as df:
                items = json.load(df)
            datasets.append({
                "filename": f.name,
                "count": len(items) if isinstance(items, list) else 0,
            })
        except Exception:
            continue

    return datasets


@app.get("/api/datasets/{filename}")
async def get_dataset_detail(filename: str):
    """Returns the QA pairs for a specific golden dataset."""
    safe_name = Path(filename).name
    target_file = ROOT / "evals" / "datasets" / safe_name

    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {str(e)}")


# ---------------------------------------------------------------------------
# API: Live Playground Query Execution
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    doc_path: str = "docs/Facebooks-Corporate-Human-Rights-Policy.pdf"
    vector_store: str = "chroma"
    top_k: int = 3
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    prompt_template: str = "default"
    use_reranker: bool = False
    temperature: float = 0.0


@app.post("/api/playground/query")
async def run_playground_query(req: QueryRequest):
    """
    Executes a live query through the decoupled RAG pipeline:
    1. Builds retriever for selected document & vector store
    2. Retrieves top-k chunks (with optional cross-encoder reranker)
    3. Generates grounded answer via LLM
    4. Measures and returns retrieval, generation, and total latency
    """
    clean_query = req.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    full_doc_path = ROOT / req.doc_path if not Path(req.doc_path).is_absolute() else Path(req.doc_path)
    if not full_doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_path}' not found.")

    try:
        # Phase 1: Retrieval
        t0 = time.perf_counter()
        retriever = build_retriever(
            doc_path=str(full_doc_path),
            vector_store_type=req.vector_store,
            chunk_size=500,
            chunk_overlap=100,
            k=req.top_k,
        )

        retrieved_texts = retrieve_contexts(
            retriever=retriever,
            query=clean_query,
            use_reranker=req.use_reranker,
            top_k=req.top_k,
        )
        t1 = time.perf_counter()

        # Phase 2: Generation
        answer = generate_answer(
            query=clean_query,
            contexts=retrieved_texts,
            provider=req.provider,
            model_name=req.model_name,
            temperature=req.temperature,
            prompt_template=req.prompt_template,
        )
        t2 = time.perf_counter()

        retrieval_ms = (t1 - t0) * 1000.0
        generation_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0

        # Structure chunks for UI display
        formatted_chunks = [
            {
                "index": i + 1,
                "content": text,
                "chars": len(text),
                "page": None,
            }
            for i, text in enumerate(retrieved_texts)
        ]

        return {
            "query": clean_query,
            "answer": answer,
            "chunks": formatted_chunks,
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": generation_ms,
            "total_latency_ms": total_ms,
            "metadata": {
                "vector_store": req.vector_store,
                "use_reranker": req.use_reranker,
                "top_k": req.top_k,
                "model": req.model_name,
                "prompt_template": req.prompt_template,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# ---------------------------------------------------------------------------
# API: Synthesize Golden Dataset
# ---------------------------------------------------------------------------
class GoldenGenRequest(BaseModel):
    doc_path: str = "docs/Facebooks-Corporate-Human-Rights-Policy.pdf"
    instruction: str = "Generate clear, representative questions and answers."
    max_goldens: int = 5
    sample_strategy: str = "stride"
    output_file: str = "custom_goldens.json"


@app.post("/api/goldens/generate")
async def trigger_golden_generation(req: GoldenGenRequest):
    """Triggers prompt-styled golden dataset synthesis."""
    full_doc_path = ROOT / req.doc_path if not Path(req.doc_path).is_absolute() else Path(req.doc_path)
    if not full_doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_path}' not found.")

    out_name = req.output_file if req.output_file.endswith(".json") else f"{req.output_file}.json"
    out_path = ROOT / "evals" / "datasets" / out_name

    try:
        goldens = generate_golden_dataset(
            doc_path=str(full_doc_path),
            instructions=[req.instruction],
            output_path=str(out_path),
            max_goldens=req.max_goldens,
            sample_strategy=req.sample_strategy,
        )
        return {
            "status": "success",
            "count": len(goldens),
            "file": out_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("   ⚡ Starting RAG Eval Suite Dashboard at http://127.0.0.1:8000")
    print("=" * 55 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

