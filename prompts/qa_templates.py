"""
QA and RAG Prompt Templates
Decoupled System Prompts and User Templates with reactive preset support.
"""

# System Prompts
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "If the answer cannot be found in the context, say \"I don't have enough information to answer that.\""
)

CONCISE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Provide a concise, direct answer to the question using ONLY the provided context. "
    "Keep your response brief and to the point. If the answer cannot be found in the context, say \"I don't have enough information to answer that.\""
)

REASONING_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "First, break down your reasoning step-by-step based strictly on the facts in the context. "
    "Then provide your final answer. If the answer cannot be found in the context, say \"I don't have enough information to answer that.\""
)

# User Prompt Template
DEFAULT_USER_TEMPLATE = """Context:
{context}

Question:
{question}

Answer:"""

# Decoupled Presets for Chat Models (System Prompt + User Prompt)
QA_PRESETS: dict[str, dict[str, str]] = {
    "default": {
        "name": "Default Grounded",
        "system": DEFAULT_SYSTEM_PROMPT,
        "user": DEFAULT_USER_TEMPLATE,
    },
    "concise": {
        "name": "Concise & Direct",
        "system": CONCISE_SYSTEM_PROMPT,
        "user": DEFAULT_USER_TEMPLATE,
    },
    "reasoning": {
        "name": "Step-by-Step Reasoning",
        "system": REASONING_SYSTEM_PROMPT,
        "user": DEFAULT_USER_TEMPLATE,
    },
}


def get_qa_preset(name: str = "default") -> dict[str, str]:
    """
    Returns the decoupled system and user prompt for a named preset.
    Defaults to 'default' preset if not found.
    """
    clean_name = (name or "default").lower().strip()
    return QA_PRESETS.get(clean_name, QA_PRESETS["default"])


def get_all_qa_presets() -> dict[str, dict[str, str]]:
    """Returns all available QA presets."""
    return QA_PRESETS



