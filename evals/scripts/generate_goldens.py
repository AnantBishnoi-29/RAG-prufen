"""
Production-Ready Golden Dataset Generator
Generates high-quality synthetic golden evaluation datasets from enterprise documents.

Features:
- Direct LLM single-call pipeline (token-efficient, high-throughput, no DeepEval engine overhead)
- Dynamic user-prompt-driven styling (custom perspective, persona, or domain instructions)
- Uniform stride sampling across large documents (e.g. 300+ page PDFs)
- Automatic checkpointing and resumability for large jobs
- Standard JSON export compatible with all downstream RAG evaluation suites
"""

import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Root directory of the repository
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from components.generator import get_llm
from components.loaders import load_documents
from components.text_splitters import split_documents
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from prompts.synthesis_prompts import SYNTHESIS_SYSTEM_PROMPT, format_synthesis_prompt

load_dotenv()


def sample_chunks(
    chunks: list[Document],
    target_count: int,
    strategy: str = "stride",
    min_chars: int = 120,
) -> list[Document]:
    """
    Filters and samples chunks from a document based on target count and strategy.

    Strategies:
    - 'stride': Uniform step across the document for comprehensive span coverage.
    - 'head_tail': Samples evenly from the beginning and end of the document.
    - 'all': Takes the first target_count valid chunks.
    """
    # 1. Filter out low-information or boilerplate chunks
    valid_chunks = [c for c in chunks if len(c.page_content.strip()) >= min_chars]
    if not valid_chunks:
        raise ValueError(f"No valid chunks found with length >= {min_chars} characters.")

    total_valid = len(valid_chunks)
    if total_valid <= target_count:
        return valid_chunks

    strategy_clean = strategy.lower().strip()

    if strategy_clean == "stride":
        # Uniformly stride across the document so all sections are represented
        step = total_valid / float(target_count)
        indices = [min(int(math.floor(i * step)), total_valid - 1) for i in range(target_count)]
        # Deduplicate while preserving order
        unique_indices = list(dict.fromkeys(indices))
        return [valid_chunks[i] for i in unique_indices]

    elif strategy_clean == "head_tail":
        half = target_count // 2
        head = valid_chunks[:half]
        tail = valid_chunks[-(target_count - half):]
        return head + tail

    elif strategy_clean == "all":
        return valid_chunks[:target_count]

    else:
        raise ValueError(f"Unsupported sampling strategy: '{strategy}'. Choose from: 'stride', 'head_tail', 'all'.")


def parse_llm_json(raw_text: str) -> dict[str, str]:
    """Extracts and parses JSON object from LLM response with cleanup."""
    text = raw_text.strip()

    # Strip markdown fences if present
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {
                "question": str(data.get("question", "")).strip(),
                "expected_answer": str(data.get("expected_answer", "")).strip(),
            }
    except json.JSONDecodeError:
        pass

    # Regex fallback if JSON decoding fails
    q_match = re.search(r'"question"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
    a_match = re.search(r'"expected_answer"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)

    question = q_match.group(1).encode().decode("unicode_escape") if q_match else ""
    expected_answer = a_match.group(1).encode().decode("unicode_escape") if a_match else ""

    if not question or not expected_answer:
        raise ValueError(f"Failed to parse question and expected_answer from LLM response: {raw_text[:200]}...")

    return {"question": question, "expected_answer": expected_answer}


def generate_golden_dataset(
    doc_path: str = str(ROOT / "docs" / "Facebooks-Corporate-Human-Rights-Policy.pdf"),
    pdf_path: str | None = None,  # alias for backward compatibility
    instructions: list[str] | str | None = None,
    output_path: str | None = None,
    output_dir: str = str(ROOT / "evals" / "datasets"),
    output_file: str = "golden_dataset",
    max_goldens: int = 10,
    max_goldens_per_context: int | None = None,  # alias for backward compatibility
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    sample_strategy: str = "stride",
    provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.3,
    enable_checkpoint: bool = True,
) -> list[dict[str, Any]]:
    """
    Generates a high-quality golden dataset driven directly by user instructions and chunks.

    Parameters:
    - doc_path: Path to source document (PDF, TXT, MD, or folder).
    - pdf_path: Optional alias for doc_path (for backward compatibility).
    - instructions: One or more user styling prompts / instructions.
    - output_path: Exact destination JSON file path (overrides output_dir / output_file).
    - output_dir: Directory to save generated dataset.
    - output_file: Base filename if output_path is not specified.
    - max_goldens: Total target number of QA pairs to generate.
    - max_goldens_per_context: Optional alias for backward compatibility.
    - chunk_size: Chunk size in characters.
    - chunk_overlap: Overlap between consecutive chunks.
    - sample_strategy: 'stride' (uniform), 'head_tail', or 'all'.
    - provider: LLM provider ('openai', 'deepseek', 'ollama').
    - model_name: Name of the model.
    - temperature: Sampling temperature.
    - enable_checkpoint: Whether to save intermediate progress and resume on failure.
    """
    actual_doc_path = pdf_path if pdf_path is not None else doc_path
    if max_goldens_per_context is not None and max_goldens == 10:
        max_goldens = max_goldens_per_context * 2

    if instructions is None:
        user_instructions = [
            "Generate a clear, realistic question and comprehensive answer testing factual understanding of this excerpt."
        ]
    elif isinstance(instructions, str):
        user_instructions = [instructions]
    else:
        user_instructions = [inst for inst in instructions if inst and inst.strip()]
        if not user_instructions:
            user_instructions = ["Generate a clear question and factual answer based strictly on the excerpt."]

    if output_path:
        final_output_path = Path(output_path).resolve()
    else:
        fname = output_file if output_file.endswith(".json") else f"{output_file}.json"
        final_output_path = (Path(output_dir) / fname).resolve()

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = final_output_path.with_suffix(".checkpoint.json")

    # 1. Check for existing checkpoint
    completed_goldens: list[dict[str, Any]] = []
    if enable_checkpoint and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as cp_file:
                completed_goldens = json.load(cp_file)
            print(f"🔄 Resuming from existing checkpoint: {len(completed_goldens)} test cases loaded.")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint ({e}); starting fresh.")
            completed_goldens = []

    if len(completed_goldens) >= max_goldens:
        print(f"✅ Checkpoint already contains target count ({len(completed_goldens)} >= {max_goldens}).")
        with open(final_output_path, "w", encoding="utf-8") as f:
            json.dump(completed_goldens[:max_goldens], f, indent=4, ensure_ascii=False)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        return completed_goldens[:max_goldens]

    # 2. Load and chunk source document
    raw_docs = load_documents(actual_doc_path)
    print(f"Splitting documents (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    chunks = split_documents(
        raw_docs,
        splitter_type="recursive",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"Total raw chunks generated: {len(chunks)}")

    # 3. Sample chunks according to strategy
    sampled = sample_chunks(
        chunks=chunks,
        target_count=max_goldens,
        strategy=sample_strategy,
    )
    print(f"Sampled {len(sampled)} chunks using strategy '{sample_strategy}'.")

    # 4. Initialize LLM
    print(f"Initializing LLM generator ({provider}: {model_name}, temp={temperature})...")
    llm = get_llm(provider=provider, model_name=model_name, temperature=temperature)

    # 5. Generate Goldens
    doc_filename = Path(actual_doc_path).name
    start_index = len(completed_goldens)

    print(f"\n🚀 Starting QA generation: {len(sampled) - start_index} pair(s) remaining...")
    print(f"Active user instruction(s) count: {len(user_instructions)}")

    for i in range(start_index, len(sampled)):
        chunk = sampled[i]
        current_instruction = user_instructions[i % len(user_instructions)]

        prompt_content = format_synthesis_prompt(
            user_instruction=current_instruction,
            context=chunk.page_content,
        )

        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content),
        ]

        try:
            response = llm.invoke(messages)
            parsed = parse_llm_json(response.content)

            golden_item: dict[str, Any] = {
                "input": parsed["question"],
                "actual_output": None,
                "expected_output": parsed["expected_answer"],
                "context": [chunk.page_content],
                "source_file": doc_filename,
                "metadata": {
                    "user_instruction": current_instruction,
                    "chunk_index": i,
                    "source_page": chunk.metadata.get("page", None),
                },
            }
            completed_goldens.append(golden_item)

            print(
                f"  [{len(completed_goldens)}/{len(sampled)}] Question: "
                f"{parsed['question'][:75]}..."
            )

            # Checkpoint after each successful pair
            if enable_checkpoint:
                with open(checkpoint_path, "w", encoding="utf-8") as cp_file:
                    json.dump(completed_goldens, cp_file, indent=4, ensure_ascii=False)

        except Exception as err:
            print(f"  ❌ Error generating golden for chunk {i}: {err}")
            continue

    # 6. Save final output dataset
    with open(final_output_path, "w", encoding="utf-8") as out_file:
        json.dump(completed_goldens, out_file, indent=4, ensure_ascii=False)

    print(f"\n🎉 Successfully generated {len(completed_goldens)} golden test case(s).")
    print(f"📁 Dataset saved to: {final_output_path}")

    # 7. Clean up checkpoint
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    return completed_goldens


if __name__ == "__main__":
    generate_golden_dataset(max_goldens=5)
