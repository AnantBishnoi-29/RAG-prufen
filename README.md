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
├── databases/                   # Centralized storage for persistent vector databases
│   └── chroma/                  # (and future: faiss, qdrant, etc.)
├── prompts/                     # Prompt templates isolated from code
│   └── qa_templates.py
├── components/                  # Modular RAG Building Blocks
│   ├── loaders.py               # Document loaders
│   ├── text_splitters.py        # Text splitters
│   ├── embeddings.py            # Embedding models
│   ├── vector_stores.py         # Vector databases & retrievers
│   ├── rerankers.py             # Cross-encoder / reranking models
│   ├── generator.py             # Answer generation & LLM interface
│   └── cost_tracker.py          # Token usage & cost tracking
├── evals/                       # Evaluation Suite
│   ├── datasets/                # Golden datasets (JSON)
│   │   └── golden_dataset.json
│   ├── datasets/                # Golden & Safety datasets (JSON)
│   │   ├── golden_dataset.json
│   │   └── safety_dataset.json
│   ├── results/                 # Exported JSON benchmark runs
│   ├── scripts/                 # Dataset generation scripts
│   │   └── generate_goldens.py
│   ├── test_generator.py        # Dedicated Faithfulness, Relevancy & Hallucination evals
│   ├── test_retriever.py        # Dedicated Context Recall & Context Precision evals
│   └── test_rag.py              # End-to-end Faithfulness & Relevancy evals
│   ├── test_generator.py        # Dedicated generator isolation evals
│   ├── test_retriever.py        # Dedicated retriever evals (Recall & Precision)
│   ├── test_generator.py        # Dedicated generator isolation evals (Faithfulness, Relevancy, Hallucination)
│   ├── test_safety.py           # Safety & robustness evals (Toxicity, Bias, Injections)
│   ├── test_ops.py              # Operations & performance evals (Latency, Tokens, Cost, Throughput)
│   └── test_rag.py              # End-to-end live RAG pipeline evals
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
- [x] **Phase 4: Generator Decoupling & Evaluation (Completed)**
- [x] **Phase 4: Generator Decoupling & Safety Evaluation (Completed)**
- [x] **Phase 4: Generator Decoupling, Safety & Ops Evaluation (Completed)**
  - Decoupled LLM factory (`get_llm`) supporting OpenAI, DeepSeek, and Ollama/Local
  - Prompt template registry (`prompts/qa_templates.py`: default, concise, reasoning)
  - Dedicated generator evaluations: **Faithfulness**, **Answer Relevancy**, and **Hallucination** (`evals/test_generator.py`) supporting both isolated ground-truth context and live retrieval mode
  - Dedicated generator isolation evaluations: **Faithfulness**, **Answer Relevancy**, and **Hallucination** (`evals/test_generator.py`)
  - Prompt template registry (`prompts/qa_templates.py`, `prompts/synthesis_prompts.py`)
  - Dedicated generator evaluations: **Faithfulness**, **Answer Relevancy**, and **Hallucination** (`evals/test_generator.py`)
  - Dedicated safety & robustness evaluations: **Toxicity**, **Bias**, and **Adversarial Injections** (`evals/test_safety.py`, `evals/datasets/safety_dataset.json`)
  - Dedicated operations & performance benchmarking: **Latency (P50/P95), Token Usage, USD Cost & Throughput** (`evals/test_ops.py`)
  - End-to-end live RAG evaluation runner (`evals/test_rag.py`)
  - **Production Golden Dataset Generator** (`evals/scripts/generate_goldens.py`):
    - Direct, high-throughput LLM pipeline (no DeepEval synthesis overhead)
    - Custom user-prompt-driven styling (supports multiple instructions/perspectives)
    - Uniform stride sampling across large enterprise documents (e.g. 300+ page PDFs)
    - Automatic checkpointing and resumability (`.checkpoint.json`)
- [ ] **Phase 5: Dashboard & Comparison Interface (Next)**
  - Streamlit web interface for interactive benchmarking and side-by-side comparison.
- [x] **Phase 5: Interactive Web Dashboard & Comparison UI (Completed)**
  - Pure HTML5, CSS3, and Vanilla JavaScript frontend (`web/index.html`, `web/static/style.css`, `web/static/app.js`)
  - FastAPI backend server (`app.py`) providing interactive playground, results browser, side-by-side run comparison, and dataset synthesis
