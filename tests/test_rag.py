import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.rag import (
    NO_CONTEXT_ANSWER,
    GenerationError,
    RAGPipeline,
    build_context,
    build_messages,
    generate_answer,
)
from app.vector_store import SearchResult, VectorRecord


def make_result(
    text: str,
    source: str = "Inception.pdf",
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        record=VectorRecord(
            vector=[1.0, 0.0],
            text=text,
            source=source,
        ),
        score=score,
    )


def make_completion(content: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                )
            )
        ]
    )


class TestPromptConstruction(unittest.TestCase):
    def test_builds_context_with_sources(self):
        results = [
            make_result("First passage."),
            make_result(
                "Second passage.",
                source="Other.pdf",
            ),
        ]

        context = build_context(results)

        self.assertIn(
            "[Passage 1 | Source: Inception.pdf]",
            context,
        )
        self.assertIn("First passage.", context)
        self.assertIn(
            "[Passage 2 | Source: Other.pdf]",
            context,
        )
        self.assertIn("Second passage.", context)

    def test_empty_results_are_rejected(self):
        with self.assertRaises(ValueError):
            build_context([])

    def test_builds_system_and_user_messages(self):
        results = [
            make_result("An idea is resilient.")
        ]

        messages = build_messages(
            question="What is resilient?",
            results=results,
        )

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(
            "An idea is resilient.",
            messages[1]["content"],
        )
        self.assertIn(
            "What is resilient?",
            messages[1]["content"],
        )


class TestAnswerGeneration(unittest.TestCase):
    def test_generates_answer(self):
        client = Mock()
        client.chat.completions.create.return_value = (
            make_completion("An idea.")
        )

        messages = [
            {
                "role": "user",
                "content": "Question",
            }
        ]

        answer = generate_answer(
            messages=messages,
            client=client,
            model="test-model",
        )

        self.assertEqual(answer, "An idea.")

        client.chat.completions.create.assert_called_once_with(
            messages=messages,
            model="test-model",
            temperature=0,
        )

    def test_provider_error_is_wrapped(self):
        client = Mock()
        client.chat.completions.create.side_effect = (
            RuntimeError("API unavailable")
        )

        with self.assertRaises(GenerationError):
            generate_answer(
                messages=[
                    {
                        "role": "user",
                        "content": "Question",
                    }
                ],
                client=client,
                model="test-model",
            )

    def test_empty_answer_is_rejected(self):
        client = Mock()
        client.chat.completions.create.return_value = (
            make_completion("   ")
        )

        with self.assertRaises(GenerationError):
            generate_answer(
                messages=[
                    {
                        "role": "user",
                        "content": "Question",
                    }
                ],
                client=client,
                model="test-model",
            )


class TestRAGPipeline(unittest.TestCase):
    def test_retrieves_and_generates_answer(self):
        retriever = Mock()
        retriever.retrieve.return_value = [
            make_result("An idea is resilient.")
        ]

        client = Mock()
        client.chat.completions.create.return_value = (
            make_completion("The parasite is an idea.")
        )

        pipeline = RAGPipeline(
            retriever=retriever,
            groq_client=client,
            groq_model="test-model",
        )

        response = pipeline.answer(
            "What is the most resilient parasite?"
        )

        self.assertEqual(
            response.answer,
            "The parasite is an idea.",
        )
        self.assertEqual(len(response.results), 1)

        retriever.retrieve.assert_called_once_with(
            "What is the most resilient parasite?"
        )

    def test_empty_retrieval_skips_generation(self):
        retriever = Mock()
        retriever.retrieve.return_value = []

        client = Mock()

        pipeline = RAGPipeline(
            retriever=retriever,
            groq_client=client,
            groq_model="test-model",
        )

        response = pipeline.answer(
            "An unanswerable question"
        )

        self.assertEqual(
            response.answer,
            NO_CONTEXT_ANSWER,
        )
        self.assertEqual(response.results, ())

        client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()