from components.cost_tracker import format_cost_summary, track_cost
from components.generator import generate_answer
from components.retriever import create_vector_store, get_retriever, retrieve_contexts
from components.retriever import build_retriever, retrieve_contexts


def run_pipeline():
    # 1. Sample knowledge base
    sample_docs = [
        "LangChain is an open-source framework designed to simplify the creation of applications using large language models.",
        "Retrieval-Augmented Generation (RAG) optimizes LLM output by referencing an authoritative external knowledge base before generating a response.",
        "Chroma is an open-source vector database designed to store and query embeddings.",
    ]
    # 1. Build vector store and retriever
    print("Indexing document into Chroma...")
    retriever = build_retriever(
        doc_path="docs/Facebooks-Corporate-Human-Rights-Policy.pdf",
        k=2,
    )

    # 2. Build vector store and retriever
    print("Indexing sample data into Chroma...")
    vector_store = create_vector_store("docs/Facebooks-Corporate-Human-Rights-Policy.pdf")
    retriever = get_retriever(vector_store, k=2)

    # 3. Retrieve relevant context
    query = "What is RAG and how does it work?"
    # 2. Retrieve relevant context
    query = "What is Meta's commitment to human rights?"
    print(f"\nQuery: {query}")
    contexts = retrieve_contexts(retriever, query)
    contexts = retrieve_contexts(retriever=retriever, query=query)
    print(f"Retrieved {len(contexts)} relevant chunks:")
    for i, chunk in enumerate(contexts, 1):
        print(f"  [{i}] {chunk}")
        preview = chunk.replace("\n", " ")[:120]
        print(f"  [{i}] {preview}...")

    # 4. Generate answer with cost tracking
    print("\nGenerating answer with OpenAI...")
    # 3. Generate answer with cost tracking
    print("\nGenerating answer with OpenAI (gpt-4o-mini)...")
    with track_cost() as cost:
        answer = generate_answer(query=query, contexts=contexts)
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


