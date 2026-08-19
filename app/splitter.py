from functools import lru_cache, partial
from typing import Callable

import nltk
import tiktoken


TOKEN_ENCODING = "cl100k_base"


@lru_cache
def get_tokenizer():
    return tiktoken.get_encoding(TOKEN_ENCODING)


@lru_cache
def get_sentence_tokenizer():
    return nltk.data.load("tokenizers/punkt/english.pickle")


def token_size(text: str) -> int:
    return len(get_tokenizer().encode(text))


def split_by_separator(text: str, separator: str) -> list[str]:
    if not text:
        return []

    pieces = text.split(separator)

    splits = [
        piece + separator
        for piece in pieces[:-1]
    ]

    if pieces[-1]:
        splits.append(pieces[-1])

    return splits


def split_sentences(text: str) -> list[str]:
    if not text:
        return []

    tokenizer = get_sentence_tokenizer()
    spans = list(tokenizer.span_tokenize(text))

    if not spans:
        return [text]

    starts = [start for start, _ in spans]

    # Preserve whitespace before the first detected sentence.
    starts[0] = 0
    starts.append(len(text))

    return [
        text[starts[index]:starts[index + 1]]
        for index in range(len(starts) - 1)
    ]


class TextSplitter:
    def __init__(self, chunk_size: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        self.chunk_size = chunk_size

        self.splitters: tuple[Callable[[str], list[str]], ...] = (
            partial(split_by_separator, separator="\n\n"),
            partial(split_by_separator, separator="\n"),
            split_sentences,
            partial(split_by_separator, separator=" "),
        )

    def _split_by_tokens(self, text: str) -> list[str]:
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(text)

        return [
            tokenizer.decode(tokens[start:start + self.chunk_size])
            for start in range(0, len(tokens), self.chunk_size)
        ]

    def _split_recursive(
        self,
        text: str,
        level: int = 0,
    ) -> list[str]:
        if not text:
            return []

        if token_size(text) <= self.chunk_size:
            return [text]

        if level >= len(self.splitters):
            return self._split_by_tokens(text)

        splits: list[str] = []

        for piece in self.splitters[level](text):
            if token_size(piece) <= self.chunk_size:
                splits.append(piece)
            else:
                splits.extend(
                    self._split_recursive(piece, level + 1)
                )

        return splits

    def _merge_splits(self, splits: list[str]) -> list[str]:
        chunks: list[str] = []
        current_chunk = ""

        for split in splits:
            proposed_chunk = current_chunk + split

            if (
                current_chunk
                and token_size(proposed_chunk) > self.chunk_size
            ):
                trimmed_chunk = current_chunk.strip()

                if trimmed_chunk:
                    chunks.append(trimmed_chunk)

                current_chunk = ""

            current_chunk += split

        trimmed_chunk = current_chunk.strip()

        if trimmed_chunk:
            chunks.append(trimmed_chunk)

        return chunks

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []

        splits = self._split_recursive(text)
        chunks = self._merge_splits(splits)

        oversized_chunks = [
            chunk
            for chunk in chunks
            if token_size(chunk) > self.chunk_size
        ]

        if oversized_chunks:
            raise RuntimeError(
                "TextSplitter produced a chunk larger than chunk_size"
            )

        return chunks

    def __call__(self, text: str) -> list[str]:
        return self.split(text)