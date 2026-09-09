"""
Prompt templates for Golden Dataset Synthesis.
Supports user-defined instructions to dynamically steer QA generation.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are an expert dataset curation engine. Your job is to extract high-quality, factually grounded question-and-answer pairs from provided document excerpts.

You must follow the user's specific instruction regarding tone, perspective, domain, or persona.
All questions and expected answers MUST be strictly supported by and answerable from the provided context excerpt. Do not invent or extrapolate information beyond the context."""

SYNTHESIS_USER_TEMPLATE = """User Instruction:
{user_instruction}

Context Excerpt:
\"\"\"
{context}
\"\"\"

Task:
Based SOLELY on the context excerpt above, generate a realistic question and a comprehensive, accurate expected answer according to the user instruction.

Respond ONLY with a valid JSON object in the following format, with no surrounding commentary or explanation:
{{
  "question": "The generated question",
  "expected_answer": "The accurate, grounded answer"
}}"""


def format_synthesis_prompt(user_instruction: str, context: str) -> str:
    """Formats the synthesis prompt with the user instruction and chunk context."""
    instruction = (
        user_instruction.strip()
        if user_instruction and user_instruction.strip()
        else "Generate a clear, representative question and answer that tests understanding of this excerpt."
    )
    return SYNTHESIS_USER_TEMPLATE.format(
        user_instruction=instruction,
        context=context.strip(),
    )

