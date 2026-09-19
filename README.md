# RAG Eval Suite

A modular testing and evaluation suite designed to benchmark, compare, and identify optimal RAG (Retrieval-Augmented Generation) configurations before committing to a production architecture.

---

## 🎯 Motivation & Purpose

Implementing a production-grade RAG pipeline requires choosing between numerous options:
- Which document loader preserves structure best?
- What chunking strategy (fixed size, recursive, semantic) and overlap yields the most relevant context?
- Which embedding model and vector database offer the best balance of speed, accuracy, and cost?
- Does adding a reranker significantly improve retrieval quality?
- Which LLM generator and prompt template minimize hallucinations and maximize faithfulness?

**RAG Eval Suite** provides a flexible testing framework so developers can experiment with different configurations, evaluate them quantitatively using standard RAG metrics, and make data-driven decisions before writing production code.

---

## 🏗️ Architecture & Component Choices

The project is structured modularly to allow swapping components seamlessly:

1. **Document Loaders**: PDF (e.g. `PyPDFDirectoryLoader`), text files, markdown, web loaders.
2. **Text Splitters**: RecursiveCharacterTextSplitter, semantic splitters, token-based splitters with configurable chunk sizes & overlap.
3. **Embeddings & Vector Stores**: OpenAI Embeddings, HuggingFace (`sentence-transformers`), Chroma DB, etc.
4. **Rerankers**: Cross-encoders, Cohere rerankers.
5. **Generators**: OpenAI (`gpt-4o-mini`), DeepSeek, local LLMs, custom prompt templates, temperature controls.

---

## 📊 Evaluation Dimensions

Using evaluation libraries like **DeepEval** (and Ragas):

- **Retriever Evaluation**:
  - Context Recall
  - Context Precision
  - Hit Rate / MRR
- **Generator Evaluation**:
  - Faithfulness / Groundedness
  - Answer Relevance
  - Hallucination Detection
- **End-to-End Evaluation**:
  - Ground-truth Answer Correctness
- **Ops & Efficiency**:
  - Retrieval & generation latency
  - Token usage & cost estimation
- **Safety & Robustness**:
  - Toxicity, bias, prompt injection vulnerability

---

## 📁 Project Structure

```text
rag_eval_suite/
├── docs/                        # Raw input documents (PDFs, text files, etc.)
│   ├── Facebooks-Corporate-Human-Rights-Policy.pdf
│   └── open_elective.pdf
├── databases/                   # Centralized storage for persistent vector databases & BM25
│   ├── chroma/
│   ├── faiss/
│   └── bm25/
├── prompts/                     # Prompt templates isolated from code
│   ├── qa_templates.py
│   └── synthesis_prompts.py
├── components/                  # Modular RAG Building Blocks
│   ├── loaders.py               # Document loaders
│   ├── text_splitters.py        # Text splitters
│   ├── embeddings.py            # Embedding models
│   ├── vector_stores.py         # Vector databases, BM25 retriever & disk caching
│   ├── rerankers.py             # Cross-encoder / reranking models
│   ├── generator.py             # Answer generation & LLM interface
│   └── cost_tracker.py          # Token usage & cost tracking
├── evals/                       # Evaluation Suite
│   ├── datasets/                # Golden & Safety datasets (JSON)
│   │   ├── golden_dataset.json
│   │   └── safety_dataset.json
│   ├── results/                 # Exported JSON benchmark runs
│   ├── scripts/                 # Dataset generation scripts
│   │   └── generate_goldens.py
│   ├── registry.py              # Metric catalog & dynamic discovery registry
│   ├── test_retriever.py        # Dedicated Context Recall & Context Precision evals
│   ├── test_generator.py        # Dedicated Faithfulness, Relevancy & Hallucination evals
│   ├── test_safety.py           # Safety & robustness evals (Toxicity, Bias, Injections)
│   ├── test_ops.py              # Operations & performance evals (Latency, Tokens, Cost, Throughput)
│   └── test_rag.py              # End-to-end live RAG pipeline evals with attached Ops metrics
├── web/                         # Interactive Web Dashboard (HTML5, CSS3, Vanilla JS)
│   ├── index.html
│   └── static/
│       ├── app.js
│       └── style.css
├── app.py                       # FastAPI backend server
└── main.py                      # Pipeline entry point
```

---

## 🗺️ Incremental Roadmap

Following the principle: **First make it work, then make it work better.**

- [x] **Phase 1: Baseline RAG Pipeline**
  - Retriever (`components/retriever.py`), generator (`components/generator.py`), runner (`main.py`)
- [x] **Phase 2: Baseline Evaluations**
  - Synthetic golden generation script (`evals/scripts/generate_goldens.py`)
  - Curated golden dataset (`evals/datasets/golden_dataset.json`)
  - Evaluation test suite using `deepeval` (`evals/test_rag.py`)
  - Validated Faithfulness & Answer Relevancy metrics
- [x] **Phase 3: Retriever Decoupling & Evaluation**
  - Decoupled loaders, splitters, embeddings, vector stores, and reranker
  - Dedicated retriever evaluations: **Context Recall** and **Context Precision** (`evals/test_retriever.py`)
- [x] **Phase 4: Generator Decoupling, Safety & Ops Evaluation**
  - Decoupled LLM factory (`get_llm`) supporting OpenAI, DeepSeek, and Ollama/Local
  - Prompt template registry (`prompts/qa_templates.py`, `prompts/synthesis_prompts.py`)
  - Dedicated generator evaluations: **Faithfulness**, **Answer Relevancy**, and **Hallucination** (`evals/test_generator.py`)
  - Dedicated safety & robustness evaluations: **Toxicity**, **Bias**, and **Adversarial Injections** (`evals/test_safety.py`, `evals/datasets/safety_dataset.json`)
  - Dedicated operations & performance benchmarking: **Latency (P50/P95), Token Usage, USD Cost & Throughput** (`evals/test_ops.py`)
  - End-to-end live RAG evaluation runner (`evals/test_rag.py`)
  - **Production Golden Dataset Generator** (`evals/scripts/generate_goldens.py`):
    - Direct, high-throughput LLM pipeline (no DeepEval synthesis overhead)
    - Dynamic prompt-driven styling (supports custom instructions/perspectives)
    - Uniform stride sampling across large enterprise documents
    - Automatic checkpointing and resumability (`.checkpoint.json`)
- [x] **Phase 5: Interactive Web Dashboard & Advanced Benchmarking**
  - Pure HTML5, CSS3, and modern Vanilla JavaScript frontend (`web/index.html`, `web/static/style.css`, `web/static/app.js`)
  - Native FastAPI backend server (`app.py`) providing:
    - **Live Playground**: Interactive querying, top-k slider, vector store switching, reranker toggle, and chunk inspection with live latency breakdown.
    - **Benchmark Results Viewer**: Scorecards, Ops stats (P50/P95 latency, tokens/sec, cost), and per-test-case inspector.
    - **Side-by-Side Comparison**: Comparative diffing of hyperparameters, metric deltas (+/- %), duration, and cost.
    - **Golden Dataset Viewer & Synthesis**: Browse datasets and trigger prompt-styled golden generation.
    - **Selective Evaluation Launcher & Metric Registry** (`evals/registry.py`): Granular scope selection (`End-to-End`, `Retriever Only`, `Generator Only`) with conditional generation bypass.
    - **Vector Index Caching & Fast Disk Loading**: Deterministic index slugs (`c_<doc>_<chunk_size>_<hash>`) mounting local vector stores in ~500ms (>95% faster, 0 tokens on hit).
    - **BM25 Sparse Keyword Retrieval & Disk Caching**: Zero-cost lexical retrieval with sub-millisecond pkl deserialization and Cross-Encoder reranking compatibility.
    - **Hybrid Search (Dense + BM25 via Reciprocal Rank Fusion)**: Concurrent retrieval fusing semantic vector search with BM25 keyword search using deterministic RRF scoring, compatible with cross-encoder reranking and dynamic UI toggles.
- [x] **Phase 6: Architectural Hardening, Universal Cost & Decoupled Controls**
  - **Dynamic Document Loader & Text Splitter Exposure**: Decoupled document loaders (`auto`, `pypdf`, `text`, `pypdf_directory`) and text splitters (`recursive`, `character`, `token`) exposed across the Web UI (Playground, Benchmark Evaluator, Golden Generator), evaluation harnesses, and cache slug hashing (`get_collection_name`) to eliminate cache collisions.
  - **Universal Cost Tracker**: Decoupled cost and token tracking using LangChain's universal `get_usage_metadata_callback()`, supporting all LLM providers (OpenAI, DeepSeek, Ollama/local).
  - **Non-Blocking Threadpool Dispatch**: Converted blocking evaluation and synthesis endpoints from `async def` to `def` in `app.py`, preventing event loop freezing.
  - **Integrated Safety Metrics**: Added `toxicity` and `bias` to `evals/registry.py` (opt-in by default) with generator requirement integration in `evals/test_rag.py` and UI scope disabling in `web/static/app.js`.
- [x] **Phase 7: Developer Prompt Visibility, Editable Templates & Chat Decoupling**
  - **Decoupled System & User Templates**: Separated system prompts and user templates into structured presets in `prompts/qa_templates.py` (`QA_PRESETS`: `Default Grounded`, `Concise & Direct`, `Step-by-Step Reasoning`, `Custom`).
  - **Chat-Native Prompt Pipeline**: Structured `components/generator.py` using `ChatPromptTemplate.from_messages([("system", ...), ("human", DEFAULT_USER_TEMPLATE)])` with customizable `system_prompt` and locked context/question data plumbing.
  - **Interactive UI Prompt Editor**: Expandable prompt editor in the Live Playground with bidirectional synchronization: selecting presets auto-populates the System Prompt textarea, and manual edits reactively switch the selector to `Custom (User Defined)`.
  - **Editable Golden Synthesizer Directive**: Exposed editable System Prompt in the Golden Dataset Generator prefilled with `SYNTHESIS_SYSTEM_PROMPT`.
  - **Full API & Eval Suite Forwarding**: Added `GET /api/prompts/templates`, updated `QueryRequest`, `GoldenGenRequest`, `EvalRunRequest`, and forwarded custom prompts into `run_rag_eval()` and run hyperparameters.


