"""
QA and RAG Prompt Templates
"""

DEFAULT_QA_PROMPT = """You are a helpful assistant. Answer the question using ONLY the provided context. If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:"""

