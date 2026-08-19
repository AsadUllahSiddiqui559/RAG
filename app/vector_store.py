import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class VectorStoreError(RuntimeError):
    """Raised when the vector store contains invalid data."""


@dataclass(frozen=True, slots=True)
class VectorRecord:
    vector: list[float]
    text: str
    source: str | None = None

    def __post_init__(self) -> None:
        try:
            vector_array = np.asarray(self.vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("vector must contain numeric values") from exc

        if vector_array.ndim != 1 or vector_array.size == 0:
            raise ValueError("vector must be a non-empty one-dimensional array")

        if not np.all(np.isfinite(vector_array)):
            raise ValueError("vector must contain only finite values")

        if np.linalg.norm(vector_array) == 0:
            raise ValueError("vector magnitude must be greater than zero")

        if not self.text.strip():
            raise ValueError("text must not be empty")

        if self.source is not None and not self.source.strip():
            raise ValueError("source must not be blank")

        object.__setattr__(self, "vector", vector_array.tolist())

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": self.vector,
            "text": self.text,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VectorRecord":
        if not isinstance(data, dict):
            raise ValueError("vector record must be a JSON object")

        if "vector" not in data:
            raise ValueError("vector record is missing 'vector'")

        if "text" not in data:
            raise ValueError("vector record is missing 'text'")

        return cls(
            vector=data["vector"],
            text=data["text"],
            source=data.get("source"),
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    record: VectorRecord
    score: float


def cosine_similarity(
    query_vector: Sequence[float],
    vectors: Sequence[Sequence[float]],
) -> np.ndarray:
    try:
        query = np.asarray(query_vector, dtype=np.float64)
        matrix = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("vectors must contain numeric values") from exc

    if query.ndim != 1 or query.size == 0:
        raise ValueError(
            "query_vector must be a non-empty one-dimensional array"
        )

    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(
            "vectors must be a non-empty two-dimensional array"
        )

    if matrix.shape[1] != query.shape[0]:
        raise ValueError(
            "query vector and stored vectors must have equal dimensions"
        )

    if not np.all(np.isfinite(query)):
        raise ValueError("query_vector must contain only finite values")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("vectors must contain only finite values")

    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(matrix, axis=1)

    if query_norm == 0:
        raise ValueError("query_vector magnitude must be greater than zero")

    if np.any(vector_norms == 0):
        raise ValueError("stored vector magnitude must be greater than zero")

    return np.dot(matrix, query) / (vector_norms * query_norm)


class VectorStore:
    def __init__(
        self,
        records: Iterable[VectorRecord] | None = None,
    ):
        self._records: list[VectorRecord] = []

        if records is not None:
            self.add_many(records)

    @property
    def records(self) -> tuple[VectorRecord, ...]:
        return tuple(self._records)

    @property
    def dimension(self) -> int | None:
        if not self._records:
            return None

        return len(self._records[0].vector)

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: VectorRecord) -> None:
        self.add_many([record])

    def add_many(self, records: Iterable[VectorRecord]) -> None:
        new_records = list(records)

        if not new_records:
            return

        if not all(
            isinstance(record, VectorRecord)
            for record in new_records
        ):
            raise TypeError("all items must be VectorRecord instances")

        expected_dimension = self.dimension

        if expected_dimension is None:
            expected_dimension = len(new_records[0].vector)

        if any(
            len(record.vector) != expected_dimension
            for record in new_records
        ):
            raise ValueError("all vectors must have equal dimensions")

        self._records.extend(new_records)

    def query(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not self._records:
            return []

        vectors = [
            record.vector
            for record in self._records
        ]

        similarities = cosine_similarity(query_vector, vectors)
        result_count = min(top_k, len(self._records))

        sorted_indices = np.argsort(similarities)[::-1][:result_count]

        return [
            SearchResult(
                record=self._records[index],
                score=float(similarities[index]),
            )
            for index in sorted_indices
        ]

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = [
            record.to_dict()
            for record in self._records
        ]

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, file_path: Path) -> "VectorStore":
        if not file_path.exists():
            raise FileNotFoundError(
                f"Vector-store file does not exist: {file_path}"
            )

        try:
            with file_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise VectorStoreError(
                f"Vector-store file contains invalid JSON: {file_path}"
            ) from exc

        if not isinstance(payload, list):
            raise VectorStoreError(
                "Vector-store JSON must contain a list"
            )

        records: list[VectorRecord] = []

        for index, item in enumerate(payload):
            try:
                records.append(VectorRecord.from_dict(item))
            except (TypeError, ValueError) as exc:
                raise VectorStoreError(
                    f"Invalid vector record at index {index}"
                ) from exc

        return cls(records)