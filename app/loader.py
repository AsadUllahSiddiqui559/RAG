from dataclasses import dataclass
from pathlib import Path

from pdfminer.high_level import extract_text

from app.config import get_settings


@dataclass(frozen=True, slots=True)
class Document:
    source: Path
    text: str


class DocumentLoadError(RuntimeError):
    """Raised when a document cannot be loaded or contains no text."""


def discover_pdf_files(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_dir}")

    if not docs_dir.is_dir():
        raise NotADirectoryError(f"Documents path is not a directory: {docs_dir}")

    return sorted(
        path
        for path in docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def load_pdf(file_path: Path) -> Document:
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {file_path}")

    if not file_path.is_file():
        raise DocumentLoadError(f"PDF path is not a file: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise DocumentLoadError(f"Unsupported file type: {file_path}")

    try:
        text = extract_text(file_path)
    except Exception as exc:
        raise DocumentLoadError(f"Could not extract text from {file_path}") from exc

    if not text.strip():
        raise DocumentLoadError(f"No text was extracted from {file_path}")

    return Document(
        source=file_path.resolve(),
        text=text,
    )


def load_documents(docs_dir: Path) -> list[Document]:
    pdf_files = discover_pdf_files(docs_dir)
    return [load_pdf(file_path) for file_path in pdf_files]


def main() -> None:
    settings = get_settings()
    documents = load_documents(settings.docs_dir)

    if not documents:
        print(f"No PDF documents found in {settings.docs_dir}")
        return

    print(f"Loaded {len(documents)} PDF document(s)")

    for document in documents:
        print(
            f"- {document.source.name}: "
            f"{len(document.text):,} characters"
        )


if __name__ == "__main__":
    main()