from components.cost_tracker import format_cost_summary, track_cost
from components.generator import generate_answer
from components.retriever import build_retriever, retrieve_contexts


def run_pipeline(
    loader_type: int | str = "auto",
    splitter_type: str = "recursive",
):
    # 1. Build vector store and retriever with instant index caching
    print("Loading retriever (Chroma)...")
    retriever = build_retriever(
        doc_path="docs/Facebooks-Corporate-Human-Rights-Policy.pdf",
        loader_type=loader_type,
        splitter_type=splitter_type,
        k=2,
    )

    # 2. Retrieve relevant context
    query = "What is Meta's commitment to human rights?"
    print(f"\nQuery: {query}")
    contexts = retrieve_contexts(retriever=retriever, query=query)
    print(f"Retrieved {len(contexts)} relevant chunks:")
    for i, chunk in enumerate(contexts, 1):
        preview = chunk.replace("\n", " ")[:120]
        print(f"  [{i}] {preview}...")

    # 3. Generate answer with cost tracking
    print("\nGenerating answer with OpenAI (gpt-4o-mini)...")
    with track_cost() as cost:
        answer = generate_answer(
            query=query,
            contexts=contexts,
            provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            prompt_template="default",
        )

    print("\nResult:")
    print(answer)
    print(f"\n[Usage] {format_cost_summary(cost)}")


if __name__ == "__main__":
    run_pipeline()


