from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pdfminer.high_level import extract_text

from app.config import get_settings
from app.embeddings import embed_texts
from app.splitter import TextSplitter
from app.vector_store import VectorRecord, VectorStore


@dataclass(frozen=True, slots=True)
class Document:
    source: Path
    text: str


@dataclass(frozen=True, slots=True)
class TextChunk:
    source: Path
    index: int
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


def split_documents(
    documents: Sequence[Document],
    splitter: Callable[[str], list[str]],
) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    for document in documents:
        document_chunks = splitter(document.text)

        for index, text in enumerate(document_chunks):
            chunks.append(
                TextChunk(
                    source=document.source,
                    index=index,
                    text=text,
                )
            )

    return chunks


def create_vector_store(
    chunks: Sequence[TextChunk],
    embeddings: Sequence[Sequence[float]],
) -> VectorStore:
    if len(chunks) != len(embeddings):
        raise ValueError(
            "chunk count and embedding count must be equal"
        )

    records = [
        VectorRecord(
            vector=list(vector),
            text=chunk.text,
            source=chunk.source.name,
        )
        for chunk, vector in zip(chunks, embeddings)
    ]

    return VectorStore(records)


def index_documents(
    docs_dir: Path,
    vector_store_path: Path,
    chunk_size: int,
    embedding_model: str,
    inference_mode: str,
) -> VectorStore:
    documents = load_documents(docs_dir)

    if not documents:
        raise DocumentLoadError(
            f"No PDF documents found in {docs_dir}"
        )

    splitter = TextSplitter(chunk_size)
    chunks = split_documents(documents, splitter)

    if not chunks:
        raise DocumentLoadError(
            "No text chunks were produced from the documents"
        )

    embeddings = embed_texts(
        texts=[chunk.text for chunk in chunks],
        model=embedding_model,
        task_type="search_document",
        inference_mode=inference_mode,
    )

    store = create_vector_store(chunks, embeddings)
    store.save(vector_store_path)

    return store


def main() -> None:
    settings = get_settings()

    print(f"Loading PDFs from: {settings.docs_dir}")
    print(f"Chunk size: {settings.chunk_size}")
    print(f"Embedding model: {settings.nomic_model}")
    print(f"Inference mode: {settings.nomic_inference_mode}")

    store = index_documents(
        docs_dir=settings.docs_dir,
        vector_store_path=settings.vector_store_path,
        chunk_size=settings.chunk_size,
        embedding_model=settings.nomic_model,
        inference_mode=settings.nomic_inference_mode,
    )

    print()
    print(f"Stored records: {len(store)}")
    print(f"Embedding dimension: {store.dimension}")
    print(f"Saved to: {settings.vector_store_path}")


if __name__ == "__main__":
    main()