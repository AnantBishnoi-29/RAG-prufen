"""
QA and RAG Prompt Templates
"""

DEFAULT_QA_PROMPT = """You are a helpful assistant. Answer the question using ONLY the provided context. If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:"""

CONCISE_QA_PROMPT = """You are a helpful assistant. Provide a concise, direct answer to the question using ONLY the provided context. Keep your response brief and to the point. If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:"""

REASONING_QA_PROMPT = """You are a helpful assistant. Answer the question using ONLY the provided context. First, break down your reasoning step-by-step based strictly on the facts in the context. Then provide your final answer. If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:"""

PROMPT_TEMPLATES: dict[str, str] = {
    "default": DEFAULT_QA_PROMPT,
    "concise": CONCISE_QA_PROMPT,
    "reasoning": REASONING_QA_PROMPT,
}


def get_prompt_template(name: str = "default") -> str:
    """
    Returns a prompt template by name. If a custom template string is passed,
    it returns the string directly.
    """
    return PROMPT_TEMPLATES.get(name, name)


