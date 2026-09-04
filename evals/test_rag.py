import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import json
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate
from components.retriever import create_vector_store, get_retriever
from components.generator import generate_answer

pdf_path = ROOT / "docs" / "Facebooks-Corporate-Human-Rights-Policy.pdf"
vector_store = create_vector_store(str(pdf_path))

retriever = get_retriever(vector_store)

dataset_path = ROOT / "evals" / "datasets" / "golden_dataset.json"
with open(dataset_path, encoding="utf-8") as f:
    goldens = json.load(f)

test_cases = []

for g in goldens:

    docs = retriever.invoke(g["input"])
    context = [doc.page_content for doc in docs]
    answer = generate_answer(g["input"],context)

    test_cases.append(
        LLMTestCase(
            input=g["input"],
            actual_output=answer,
            retrieval_context=context
        )
    )
    



metrics = [AnswerRelevancyMetric(threshold=0.7,model="gpt-4o-mini",          include_reason=False)
           ,FaithfulnessMetric(threshold=0.7,model="gpt-4o-mini",include_reason=False)]

evaluate(test_cases=test_cases, 
         metrics=metrics
         )