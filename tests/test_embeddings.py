import unittest
from unittest.mock import Mock

from app.embeddings import EmbeddingError, embed_texts


class TestEmbedTexts(unittest.TestCase):
    def test_returns_valid_embeddings(self):
        provider = Mock(
            return_value={
                "embeddings": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            }
        )

        embeddings = embed_texts(
            texts=["First text", "Second text"],
            model="test-model",
            task_type="search_document",
            inference_mode="local",
            embed_function=provider,
        )

        self.assertEqual(
            embeddings,
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        provider.assert_called_once_with(
            texts=["First text", "Second text"],
            model="test-model",
            task_type="search_document",
            inference_mode="local",
        )

    def test_empty_text_collection_is_rejected(self):
        with self.assertRaises(EmbeddingError):
            embed_texts(
                texts=[],
                model="test-model",
                task_type="search_document",
                inference_mode="local",
                embed_function=Mock(),
            )

    def test_blank_text_is_rejected(self):
        with self.assertRaises(EmbeddingError):
            embed_texts(
                texts=["Valid text", "   "],
                model="test-model",
                task_type="search_document",
                inference_mode="local",
                embed_function=Mock(),
            )

    def test_mismatched_embedding_count_is_rejected(self):
        provider = Mock(
            return_value={
                "embeddings": [[1.0, 0.0]],
            }
        )

        with self.assertRaises(EmbeddingError):
            embed_texts(
                texts=["First text", "Second text"],
                model="test-model",
                task_type="search_document",
                inference_mode="local",
                embed_function=provider,
            )

    def test_mixed_dimensions_are_rejected(self):
        provider = Mock(
            return_value={
                "embeddings": [
                    [1.0, 0.0],
                    [1.0, 0.0, 2.0],
                ],
            }
        )

        with self.assertRaises(EmbeddingError):
            embed_texts(
                texts=["First text", "Second text"],
                model="test-model",
                task_type="search_document",
                inference_mode="local",
                embed_function=provider,
            )

    def test_zero_vector_is_rejected(self):
        provider = Mock(
            return_value={
                "embeddings": [[0.0, 0.0]],
            }
        )

        with self.assertRaises(EmbeddingError):
            embed_texts(
                texts=["Text"],
                model="test-model",
                task_type="search_document",
                inference_mode="local",
                embed_function=provider,
            )

    def test_provider_error_is_wrapped(self):
        provider = Mock(
            side_effect=RuntimeError("Provider unavailable")
        )

        with self.assertRaises(EmbeddingError) as context:
            embed_texts(
                texts=["Text"],
                model="test-model",
                task_type="search_document",
                inference_mode="local",
                embed_function=provider,
            )

        self.assertIsInstance(
            context.exception.__cause__,
            RuntimeError,
        )


if __name__ == "__main__":
    unittest.main()