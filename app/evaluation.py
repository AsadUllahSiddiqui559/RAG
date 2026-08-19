import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, get_settings
from app.retriever import Retriever
from app.vector_store import SearchResult, VectorStore


DEFAULT_EVALUATION_PATH = (
    PROJECT_ROOT / "data" / "evaluation_questions.json"
)


class EvaluationError(RuntimeError):
    """Raised when evaluation data is invalid."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_phrases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")

        if not self.question.strip():
            raise ValueError("question must not be empty")

        if not self.expected_phrases:
            raise ValueError(
                "expected_phrases must not be empty"
            )

        if any(
            not phrase.strip()
            for phrase in self.expected_phrases
        ):
            raise ValueError(
                "expected phrases must not be blank"
            )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case: EvaluationCase
    first_relevant_rank: int | None
    retrieved_count: int

    @property
    def hit(self) -> bool:
        return self.first_relevant_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        if self.first_relevant_rank is None:
            return 0.0

        return 1.0 / self.first_relevant_rank


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    results: tuple[EvaluationResult, ...]

    @property
    def hit_rate(self) -> float:
        if not self.results:
            return 0.0

        hits = sum(result.hit for result in self.results)
        return hits / len(self.results)

    @property
    def mean_reciprocal_rank(self) -> float:
        if not self.results:
            return 0.0

        total = sum(
            result.reciprocal_rank
            for result in self.results
        )

        return total / len(self.results)


def load_evaluation_cases(
    file_path: Path,
) -> list[EvaluationCase]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Evaluation file does not exist: {file_path}"
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            "Evaluation file contains invalid JSON"
        ) from exc

    if not isinstance(payload, list):
        raise EvaluationError(
            "Evaluation JSON must contain a list"
        )

    cases: list[EvaluationCase] = []

    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise EvaluationError(
                f"Evaluation case {index} must be an object"
            )

        try:
            case = EvaluationCase(
                case_id=item["id"],
                question=item["question"],
                expected_phrases=tuple(
                    item["expected_phrases"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError(
                f"Invalid evaluation case at index {index}"
            ) from exc

        cases.append(case)

    if not cases:
        raise EvaluationError(
            "Evaluation file must contain at least one case"
        )

    return cases


def find_first_relevant_rank(
    results: Sequence[SearchResult],
    expected_phrases: Sequence[str],
) -> int | None:
    normalized_phrases = [
        phrase.casefold()
        for phrase in expected_phrases
    ]

    for rank, result in enumerate(results, start=1):
        normalized_text = result.record.text.casefold()

        if any(
            phrase in normalized_text
            for phrase in normalized_phrases
        ):
            return rank

    return None


def evaluate_retrieval(
    retriever: Retriever,
    cases: Sequence[EvaluationCase],
) -> EvaluationReport:
    results: list[EvaluationResult] = []

    for case in cases:
        retrieved = retriever.retrieve(case.question)

        first_relevant_rank = find_first_relevant_rank(
            results=retrieved,
            expected_phrases=case.expected_phrases,
        )

        results.append(
            EvaluationResult(
                case=case,
                first_relevant_rank=first_relevant_rank,
                retrieved_count=len(retrieved),
            )
        )

    return EvaluationReport(results=tuple(results))


def main() -> None:
    settings = get_settings()
    cases = load_evaluation_cases(
        DEFAULT_EVALUATION_PATH
    )

    vector_store = VectorStore.load(
        settings.vector_store_path
    )

    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=settings.nomic_model,
        inference_mode=settings.nomic_inference_mode,
        top_k=settings.top_k,
    )

    print(f"Evaluation cases: {len(cases)}")
    print(f"Retrieval top_k: {settings.top_k}")
    print()

    report = evaluate_retrieval(
        retriever=retriever,
        cases=cases,
    )

    for result in report.results:
        if result.hit:
            status = (
                f"HIT at rank "
                f"{result.first_relevant_rank}"
            )
        else:
            status = "MISS"

        print(
            f"[{status}] "
            f"{result.case.case_id}: "
            f"{result.case.question}"
        )

    print()
    print(f"Hit rate: {report.hit_rate:.2%}")
    print(
        "Mean reciprocal rank: "
        f"{report.mean_reciprocal_rank:.4f}"
    )


if __name__ == "__main__":
    main()