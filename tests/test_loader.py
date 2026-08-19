import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.loader import (
    DocumentLoadError,
    discover_pdf_files,
    load_documents,
    load_pdf,
)


class TestDocumentDiscovery(unittest.TestCase):
    def test_discovers_pdf_files_in_sorted_order(self):
        with TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir)

            (docs_dir / "second.pdf").touch()
            (docs_dir / "first.PDF").touch()
            (docs_dir / "notes.txt").touch()

            files = discover_pdf_files(docs_dir)

            self.assertEqual(
                [path.name for path in files],
                ["first.PDF", "second.pdf"],
            )

    def test_missing_directory_is_rejected(self):
        missing_directory = Path("/directory/that/does/not/exist")

        with self.assertRaises(FileNotFoundError):
            discover_pdf_files(missing_directory)

    def test_file_path_is_rejected_as_directory(self):
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "document.pdf"
            file_path.touch()

            with self.assertRaises(NotADirectoryError):
                discover_pdf_files(file_path)


class TestPDFLoading(unittest.TestCase):
    @patch("app.loader.extract_text")
    def test_loads_pdf_text(self, mock_extract_text):
        mock_extract_text.return_value = "Extracted screenplay text."

        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "screenplay.pdf"
            pdf_path.touch()

            document = load_pdf(pdf_path)

        self.assertEqual(document.text, "Extracted screenplay text.")
        self.assertEqual(document.source.name, "screenplay.pdf")
        mock_extract_text.assert_called_once()

    @patch("app.loader.extract_text")
    def test_empty_pdf_text_is_rejected(self, mock_extract_text):
        mock_extract_text.return_value = "   \n\n"

        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "empty.pdf"
            pdf_path.touch()

            with self.assertRaises(DocumentLoadError):
                load_pdf(pdf_path)

    def test_non_pdf_file_is_rejected(self):
        with TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "notes.txt"
            text_path.touch()

            with self.assertRaises(DocumentLoadError):
                load_pdf(text_path)

    @patch("app.loader.extract_text")
    def test_loads_multiple_documents(self, mock_extract_text):
        mock_extract_text.side_effect = [
            "First document.",
            "Second document.",
        ]

        with TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir)
            (docs_dir / "first.pdf").touch()
            (docs_dir / "second.pdf").touch()

            documents = load_documents(docs_dir)

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].source.name, "first.pdf")
        self.assertEqual(documents[0].text, "First document.")
        self.assertEqual(documents[1].source.name, "second.pdf")
        self.assertEqual(documents[1].text, "Second document.")


if __name__ == "__main__":
    unittest.main()