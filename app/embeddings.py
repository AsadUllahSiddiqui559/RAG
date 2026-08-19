from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np


EmbeddingTask = Literal["search_document", "search_query"]
InferenceMode = Literal["local", "remote"]
EmbedFunction = Callable[..., Mapping[str, Any]]


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails or returns invalid data."""


def _nomic_text_embedder(**kwargs: Any) -> Mapping[str, Any]:
    # Import lazily so tests do not initialize the local model.
    from nomic import embed

    return embed.text(**kwargs)


def embed_texts(
    texts: Sequence[str],
    model: str,
    task_type: EmbeddingTask,
    inference_mode: InferenceMode,
    embed_function: EmbedFunction | None = None,
) -> list[list[float]]:
    text_list = list(texts)

    if not text_list:
        raise EmbeddingError("at least one text is required")

    if any(not isinstance(text, str) or not text.strip() for text in text_list):
        raise EmbeddingError("all texts must be non-empty strings")

    if not model.strip():
        raise EmbeddingError("embedding model must not be empty")

    provider = embed_function or _nomic_text_embedder

    try:
        response = provider(
            texts=text_list,
            model=model,
            task_type=task_type,
            inference_mode=inference_mode,
        )
    except Exception as exc:
        raise EmbeddingError("embedding provider failed") from exc

    if not isinstance(response, Mapping):
        raise EmbeddingError("embedding provider returned an invalid response")

    raw_embeddings = response.get("embeddings")

    if raw_embeddings is None:
        raise EmbeddingError("embedding response is missing 'embeddings'")

    try:
        embedding_matrix = np.asarray(
            raw_embeddings,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(
            "embeddings must contain numeric values"
        ) from exc

    if embedding_matrix.ndim != 2:
        raise EmbeddingError(
            "embeddings must be a two-dimensional array"
        )

    if embedding_matrix.shape[0] != len(text_list):
        raise EmbeddingError(
            "embedding count does not match text count"
        )

    if embedding_matrix.shape[1] == 0:
        raise EmbeddingError("embedding vectors must not be empty")

    if not np.all(np.isfinite(embedding_matrix)):
        raise EmbeddingError(
            "embeddings must contain only finite values"
        )

    vector_norms = np.linalg.norm(embedding_matrix, axis=1)

    if np.any(vector_norms == 0):
        raise EmbeddingError(
            "embedding vectors must have non-zero magnitude"
        )

    return embedding_matrix.tolist()