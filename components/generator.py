from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from prompts import get_prompt_template

load_dotenv()


def get_llm(
    provider: str = "openai",
    model_name: str | None = None,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    """
    Factory function for instantiating LLM chat models across providers.
    Supported providers: 'openai' (default), 'deepseek', 'ollama'.
    """
    provider_clean = provider.lower().strip()

    if provider_clean == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
            **kwargs,
        )

    elif provider_clean == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model=model_name or "deepseek-chat",
            temperature=temperature,
            **kwargs,
        )

    elif provider_clean in ("ollama", "local"):
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=model_name or "llama3",
            temperature=temperature,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        f"Choose from: 'openai', 'deepseek', 'ollama'."
    )


def generate_answer(
    query: str,
    contexts: list[str],
    llm: BaseChatModel | None = None,
    provider: str = "openai",
    model_name: str | None = None,
    temperature: float = 0.0,
    prompt_template: str = "default",
    **kwargs,
) -> str:
    """
    Generates an answer using the provided query and retrieved contexts.
    Can accept a pre-built LLM instance or build one using provider, model_name, and temperature.
    """
    if llm is None:
        llm = get_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            **kwargs,
        )

    raw_template = get_prompt_template(prompt_template)
    combined_context = "\n\n".join(contexts)

    prompt = ChatPromptTemplate.from_template(raw_template)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": combined_context,
        "question": query,
    })

