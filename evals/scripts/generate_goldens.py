from pathlib import Path
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer

ROOT = Path(__file__).resolve().parent.parent.parent

# Load environment variables (OPENAI_API_KEY)
load_dotenv()


def generate_golden_dataset(
    pdf_path: str = str(ROOT / "docs" / "Facebooks-Corporate-Human-Rights-Policy.pdf"),
    output_dir: str = str(ROOT / "evals" / "datasets"),
    output_file: str = "golden_dataset",
    max_goldens_per_context: int = 4,
):
    """
    Generates synthetic golden QA pairs from a PDF using DeepEval Synthesizer
    and saves them as a JSON file for manual review/editing.
    """
    print(f"Initializing DeepEval Synthesizer with gpt-4o-mini...")
    synthesizer = Synthesizer(model="gpt-4o-mini")

    print(f"Generating golden test cases from: {pdf_path}...")
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[pdf_path],
        max_goldens_per_context=max_goldens_per_context,
        include_expected_output=True,
    )

    print(f"Successfully generated {len(goldens)} golden test cases.")

    # Save to JSON
    saved_path = synthesizer.save_as(
        file_type="json",
        directory=output_dir,
        file_name=output_file,
    )
    print(f"Golden dataset saved to: {saved_path}")
    print("You can now open this JSON file to review and edit the QA pairs.")
    return goldens


if __name__ == "__main__":
    generate_golden_dataset()

