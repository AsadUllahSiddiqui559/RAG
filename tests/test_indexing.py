import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from app.loader import (
    Document,
    TextChunk,
    create_vector_store,
    index_documents,
    split_documents,
)
from app.vector_store import VectorStore


class TestDocumentChunking(unittest.TestCase):
    def test_splits_documents_and_preserves_sources(self):
        documents = [
            Document(
                source=Path("/documents/first.pdf"),
                text="First document",
            ),
            Document(
                source=Path("/documents/second.pdf"),
                text="Second document",
            ),
        ]

        splitter = Mock(
            side_effect=[
                ["First A", "First B"],
                ["Second A"],
            ]
        )

        chunks = split_documents(documents, splitter)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["First A", "First B", "Second A"],
        )
        self.assertEqual(
            [chunk.index for chunk in chunks],
            [0, 1, 0],
        )
        self.assertEqual(
            [chunk.source.name for chunk in chunks],
            ["first.pdf", "first.pdf", "second.pdf"],
        )

    def test_empty_document_collection_returns_no_chunks(self):
        splitter = Mock()

        chunks = split_documents([], splitter)

        self.assertEqual(chunks, [])
        splitter.assert_not_called()


class TestVectorStoreCreation(unittest.TestCase):
    def test_creates_records_from_chunks_and_embeddings(self):
        chunks = [
            TextChunk(
                source=Path("/documents/inception.pdf"),
                index=0,
                text="First chunk",
            ),
            TextChunk(
                source=Path("/documents/inception.pdf"),
                index=1,
                text="Second chunk",
            ),
        ]

        store = create_vector_store(
            chunks=chunks,
            embeddings=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        self.assertEqual(len(store), 2)
        self.assertEqual(store.dimension, 2)
        self.assertEqual(
            store.records[0].text,
            "First chunk",
        )
        self.assertEqual(
            store.records[0].source,
            "inception.pdf",
        )

    def test_mismatched_counts_are_rejected(self):
        chunks = [
            TextChunk(
                source=Path("inception.pdf"),
                index=0,
                text="Only chunk",
            )
        ]

        with self.assertRaises(ValueError):
            create_vector_store(
                chunks=chunks,
                embeddings=[],
            )


class TestIndexDocuments(unittest.TestCase):
    @patch("app.loader.embed_texts")
    @patch("app.loader.load_documents")
    def test_indexes_and_saves_documents(
        self,
        mock_load_documents,
        mock_embed_texts,
    ):
        mock_load_documents.return_value = [
            Document(
                source=Path("/documents/first.pdf"),
                text="First document text.",
            ),
            Document(
                source=Path("/documents/second.pdf"),
                text="Second document text.",
            ),
        ]

        mock_embed_texts.return_value = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "vector_store.json"

            store = index_documents(
                docs_dir=Path("/documents"),
                vector_store_path=output_path,
                chunk_size=512,
                embedding_model="test-model",
                inference_mode="local",
            )

            loaded_store = VectorStore.load(output_path)

        self.assertEqual(len(store), 2)
        self.assertEqual(loaded_store.records, store.records)

        mock_embed_texts.assert_called_once_with(
            texts=[
                "First document text.",
                "Second document text.",
            ],
            model="test-model",
            task_type="search_document",
            inference_mode="local",
        )


if __name__ == "__main__":
    unittest.main()