import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from app.vector_store import (
    VectorRecord,
    VectorStore,
    VectorStoreError,
    cosine_similarity,
)


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        scores = cosine_similarity(
            [1.0, 0.0],
            [[1.0, 0.0]],
        )

        np.testing.assert_allclose(scores, [1.0])

    def test_orthogonal_vectors_score_zero(self):
        scores = cosine_similarity(
            [1.0, 0.0],
            [[0.0, 1.0]],
        )

        np.testing.assert_allclose(scores, [0.0])

    def test_opposite_vectors_score_negative_one(self):
        scores = cosine_similarity(
            [1.0, 0.0],
            [[-1.0, 0.0]],
        )

        np.testing.assert_allclose(scores, [-1.0])

    def test_multiple_vectors_return_multiple_scores(self):
        scores = cosine_similarity(
            [1.0, 0.0],
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
        )

        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity(
                [1.0, 0.0],
                [[1.0, 0.0, 0.0]],
            )

    def test_zero_query_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity(
                [0.0, 0.0],
                [[1.0, 0.0]],
            )

    def test_zero_stored_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity(
                [1.0, 0.0],
                [[0.0, 0.0]],
            )


class TestVectorRecord(unittest.TestCase):
    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            VectorRecord(
                vector=[1.0, 0.0],
                text="   ",
            )

    def test_empty_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            VectorRecord(
                vector=[],
                text="Example",
            )

    def test_non_finite_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            VectorRecord(
                vector=[1.0, float("nan")],
                text="Example",
            )


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.right = VectorRecord(
            vector=[1.0, 0.0],
            text="Points right",
            source="right.txt",
        )
        self.diagonal = VectorRecord(
            vector=[1.0, 1.0],
            text="Points diagonally",
            source="diagonal.txt",
        )
        self.up = VectorRecord(
            vector=[0.0, 1.0],
            text="Points up",
            source="up.txt",
        )

    def test_adds_records(self):
        store = VectorStore()

        store.add(self.right)
        store.add_many([self.diagonal, self.up])

        self.assertEqual(len(store), 3)
        self.assertEqual(store.dimension, 2)

    def test_rejects_mixed_dimensions(self):
        store = VectorStore([self.right])

        incompatible = VectorRecord(
            vector=[1.0, 0.0, 0.0],
            text="Three dimensions",
        )

        with self.assertRaises(ValueError):
            store.add(incompatible)

    def test_query_returns_results_in_descending_order(self):
        store = VectorStore([
            self.up,
            self.diagonal,
            self.right,
        ])

        results = store.query(
            query_vector=[1.0, 0.0],
            top_k=3,
        )

        self.assertEqual(
            [result.record.text for result in results],
            [
                "Points right",
                "Points diagonally",
                "Points up",
            ],
        )

        self.assertGreaterEqual(
            results[0].score,
            results[1].score,
        )
        self.assertGreaterEqual(
            results[1].score,
            results[2].score,
        )

    def test_top_k_is_limited_to_store_size(self):
        store = VectorStore([self.right, self.up])

        results = store.query(
            query_vector=[1.0, 0.0],
            top_k=10,
        )

        self.assertEqual(len(results), 2)

    def test_empty_store_returns_no_results(self):
        store = VectorStore()

        results = store.query(
            query_vector=[1.0, 0.0],
            top_k=5,
        )

        self.assertEqual(results, [])

    def test_invalid_top_k_is_rejected(self):
        store = VectorStore([self.right])

        with self.assertRaises(ValueError):
            store.query(
                query_vector=[1.0, 0.0],
                top_k=0,
            )

    def test_save_and_load_round_trip(self):
        store = VectorStore([
            self.right,
            self.diagonal,
            self.up,
        ])

        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "vector_store.json"

            store.save(file_path)
            loaded_store = VectorStore.load(file_path)

        self.assertEqual(loaded_store.records, store.records)
        self.assertEqual(loaded_store.dimension, 2)

    def test_invalid_json_is_rejected(self):
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "invalid.json"
            file_path.write_text(
                "{invalid-json",
                encoding="utf-8",
            )

            with self.assertRaises(VectorStoreError):
                VectorStore.load(file_path)


if __name__ == "__main__":
    unittest.main()