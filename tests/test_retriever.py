import unittest
from unittest.mock import Mock

from app.retriever import Retriever
from app.vector_store import VectorRecord, VectorStore


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.vector_store = VectorStore([
            VectorRecord(
                vector=[1.0, 0.0],
                text="The answer about dreams.",
                source="dreams.pdf",
            ),
            VectorRecord(
                vector=[1.0, 1.0],
                text="A partially related passage.",
                source="related.pdf",
            ),
            VectorRecord(
                vector=[0.0, 1.0],
                text="An unrelated passage.",
                source="unrelated.pdf",
            ),
        ])

    def test_retrieves_ranked_results_using_query_task(self):
        provider = Mock(
            return_value={
                "embeddings": [[1.0, 0.0]],
            }
        )

        retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model="test-model",
            inference_mode="local",
            top_k=3,
            embed_function=provider,
        )

        results = retriever.retrieve(
            "  What happens in dreams?  "
        )

        self.assertEqual(
            [result.record.text for result in results],
            [
                "The answer about dreams.",
                "A partially related passage.",
                "An unrelated passage.",
            ],
        )

        provider.assert_called_once_with(
            texts=["What happens in dreams?"],
            model="test-model",
            task_type="search_query",
            inference_mode="local",
        )

    def test_blank_question_is_rejected(self):
        provider = Mock()

        retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model="test-model",
            inference_mode="local",
            embed_function=provider,
        )

        with self.assertRaises(ValueError):
            retriever.retrieve("   ")

        provider.assert_not_called()

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            Retriever(
                vector_store=self.vector_store,
                embedding_model="test-model",
                inference_mode="local",
                top_k=0,
            )

    def test_empty_store_returns_no_results_without_embedding(self):
        provider = Mock()

        retriever = Retriever(
            vector_store=VectorStore(),
            embedding_model="test-model",
            inference_mode="local",
            embed_function=provider,
        )

        results = retriever.retrieve("A valid question")

        self.assertEqual(results, [])
        provider.assert_not_called()

    def test_dimension_mismatch_is_rejected(self):
        provider = Mock(
            return_value={
                "embeddings": [[1.0, 0.0, 0.0]],
            }
        )

        retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model="test-model",
            inference_mode="local",
            embed_function=provider,
        )

        with self.assertRaises(ValueError):
            retriever.retrieve("Question")


if __name__ == "__main__":
    unittest.main()