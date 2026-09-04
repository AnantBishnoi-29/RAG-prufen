from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompts import DEFAULT_QA_PROMPT

load_dotenv()


def generate_answer(
    query: str,
    contexts: list[str],
    prompt_template: str = DEFAULT_QA_PROMPT,
) -> str:
    """
    Generates an answer using the provided query and retrieved contexts.
    """
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    
    combined_context = "\n\n".join(contexts)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": combined_context,
        "question": query,
    })

