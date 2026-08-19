from app.config import get_settings
from app.embeddings import (
    EmbedFunction,
    InferenceMode,
    embed_texts,
)
from app.vector_store import SearchResult, VectorStore


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: str,
        inference_mode: InferenceMode,
        top_k: int = 5,
        embed_function: EmbedFunction | None = None,
    ):
        if not embedding_model.strip():
            raise ValueError("embedding_model must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.inference_mode = inference_mode
        self.top_k = top_k
        self.embed_function = embed_function

    def retrieve(self, question: str) -> list[SearchResult]:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("question must not be empty")

        if len(self.vector_store) == 0:
            return []

        query_embeddings = embed_texts(
            texts=[normalized_question],
            model=self.embedding_model,
            task_type="search_query",
            inference_mode=self.inference_mode,
            embed_function=self.embed_function,
        )

        query_vector = query_embeddings[0]

        return self.vector_store.query(
            query_vector=query_vector,
            top_k=self.top_k,
        )


def main() -> None:
    settings = get_settings()
    vector_store = VectorStore.load(settings.vector_store_path)

    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=settings.nomic_model,
        inference_mode=settings.nomic_inference_mode,
        top_k=settings.top_k,
    )

    print(f"Loaded {len(vector_store)} vector records")
    print(f"Embedding dimension: {vector_store.dimension}")
    print()

    question = input("Question: ").strip()
    results = retriever.retrieve(question)

    if not results:
        print("No relevant chunks were found.")
        return

    print()
    print(f"Top {len(results)} retrieved chunks:")

    for rank, result in enumerate(results, start=1):
        print()
        print("=" * 80)
        print(
            f"Rank: {rank} | "
            f"Score: {result.score:.4f} | "
            f"Source: {result.record.source}"
        )
        print("-" * 80)
        print(result.record.text)


if __name__ == "__main__":
    main()