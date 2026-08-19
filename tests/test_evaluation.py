import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from app.evaluation import (
    EvaluationCase,
    EvaluationError,
    evaluate_retrieval,
    find_first_relevant_rank,
    load_evaluation_cases,
)
from app.vector_store import SearchResult, VectorRecord


def make_result(
    text: str,
    score: float,
) -> SearchResult:
    return SearchResult(
        record=VectorRecord(
            vector=[1.0, 0.0],
            text=text,
            source="Inception.pdf",
        ),
        score=score,
    )


class TestEvaluationLoading(unittest.TestCase):
    def test_loads_evaluation_cases(self):
        payload = [
            {
                "id": "example",
                "question": "Example question?",
                "expected_phrases": ["expected answer"],
            }
        ]

        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "cases.json"
            file_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            cases = load_evaluation_cases(file_path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "example")
        self.assertEqual(
            cases[0].expected_phrases,
            ("expected answer",),
        )

    def test_invalid_payload_is_rejected(self):
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "cases.json"
            file_path.write_text(
                json.dumps({"not": "a list"}),
                encoding="utf-8",
            )

            with self.assertRaises(EvaluationError):
                load_evaluation_cases(file_path)


class TestRelevanceMatching(unittest.TestCase):
    def test_finds_first_relevant_rank(self):
        results = [
            make_result("Unrelated passage.", 0.9),
            make_result("The answer is an idea.", 0.8),
        ]

        rank = find_first_relevant_rank(
            results=results,
            expected_phrases=["an idea"],
        )

        self.assertEqual(rank, 2)

    def test_returns_none_for_a_miss(self):
        results = [
            make_result("Unrelated passage.", 0.9)
        ]

        rank = find_first_relevant_rank(
            results=results,
            expected_phrases=["missing answer"],
        )

        self.assertIsNone(rank)


class TestRetrievalEvaluation(unittest.TestCase):
    def test_calculates_hit_rate_and_mrr(self):
        cases = [
            EvaluationCase(
                case_id="first",
                question="First question?",
                expected_phrases=("first answer",),
            ),
            EvaluationCase(
                case_id="second",
                question="Second question?",
                expected_phrases=("second answer",),
            ),
            EvaluationCase(
                case_id="third",
                question="Third question?",
                expected_phrases=("third answer",),
            ),
        ]

        retriever = Mock()
        retriever.retrieve.side_effect = [
            [
                make_result("The first answer.", 0.9),
            ],
            [
                make_result("Unrelated.", 0.9),
                make_result("The second answer.", 0.8),
            ],
            [
                make_result("Still unrelated.", 0.9),
            ],
        ]

        report = evaluate_retrieval(
            retriever=retriever,
            cases=cases,
        )

        self.assertEqual(len(report.results), 3)
        self.assertAlmostEqual(
            report.hit_rate,
            2 / 3,
        )
        self.assertAlmostEqual(
            report.mean_reciprocal_rank,
            0.5,
        )


if __name__ == "__main__":
    unittest.main()