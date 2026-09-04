from components.generator import generate_answer
from components.retriever import create_vector_store, get_retriever, retrieve_contexts


def run_pipeline():
    # 1. Sample knowledge base
    sample_docs = [
        "LangChain is an open-source framework designed to simplify the creation of applications using large language models.",
        "Retrieval-Augmented Generation (RAG) optimizes LLM output by referencing an authoritative external knowledge base before generating a response.",
        "Chroma is an open-source vector database designed to store and query embeddings.",
    ]

    # 2. Build vector store and retriever
    print("Indexing sample data into Chroma...")
    vector_store = create_vector_store("docs/Facebooks-Corporate-Human-Rights-Policy.pdf")
    retriever = get_retriever(vector_store, k=2)

    # 3. Retrieve relevant context
    query = "What is RAG and how does it work?"
    print(f"\nQuery: {query}")
    contexts = retrieve_contexts(retriever, query)
    print(f"Retrieved {len(contexts)} relevant chunks:")
    for i, chunk in enumerate(contexts, 1):
        print(f"  [{i}] {chunk}")

    # 4. Generate answer
    print("\nGenerating answer with OpenAI...")
    answer = generate_answer(query=query, contexts=contexts)
    print("\nResult:")
    print(answer)


if __name__ == "__main__":
    run_pipeline()

