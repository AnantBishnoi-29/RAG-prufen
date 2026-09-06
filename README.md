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
│   ├── results/                 # Exported JSON benchmark runs
│   ├── scripts/                 # Dataset generation scripts
│   │   └── generate_goldens.py
│   ├── test_retriever.py        # Context Recall & Context Precision evals
│   └── test_rag.py              # End-to-end Faithfulness & Relevancy evals
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
- [x] **Phase 3: Retriever Decoupling & Evaluation (Completed)**
  - Decoupled loaders, splitters, embeddings, vector stores, and reranker
  - Dedicated retriever evaluations: **Context Recall** and **Context Precision** (`evals/test_retriever.py`)
- [ ] **Phase 4: Generator Decoupling & Evaluation (Next)**
  - Modularize LLM providers, prompt templates (`prompts/`), temperature controls
  - Generator evaluations: Faithfulness, Answer Relevancy, Hallucination
- [ ] **Phase 5: Dashboard & Comparison Interface**
  - Streamlit web interface for interactive benchmarking and side-by-side comparison.
