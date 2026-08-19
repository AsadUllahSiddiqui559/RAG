import unittest

from app.splitter import (
    TextSplitter,
    split_by_separator,
    split_sentences,
    token_size,
)


class TestTokenSize(unittest.TestCase):
    def test_empty_text_has_zero_tokens(self):
        self.assertEqual(token_size(""), 0)

    def test_longer_text_has_more_tokens(self):
        short_size = token_size("Hello")
        long_size = token_size(
            "Hello, this is a considerably longer sentence."
        )

        self.assertGreater(long_size, short_size)

    def test_token_size_is_deterministic(self):
        text = "The same text should produce the same token count."

        self.assertEqual(token_size(text), token_size(text))


class TestSeparatorSplitting(unittest.TestCase):
    def test_paragraph_separator_is_preserved(self):
        text = "First paragraph.\n\nSecond paragraph."

        splits = split_by_separator(text, "\n\n")

        self.assertEqual(
            splits,
            ["First paragraph.\n\n", "Second paragraph."],
        )
        self.assertEqual("".join(splits), text)

    def test_line_separator_is_preserved(self):
        text = "First line.\nSecond line."

        splits = split_by_separator(text, "\n")

        self.assertEqual(
            splits,
            ["First line.\n", "Second line."],
        )
        self.assertEqual("".join(splits), text)

    def test_empty_text_returns_no_splits(self):
        self.assertEqual(split_by_separator("", "\n"), [])


class TestSentenceSplitting(unittest.TestCase):
    def test_splits_multiple_sentences(self):
        text = "First sentence. Second sentence! Third sentence?"

        splits = split_sentences(text)

        self.assertEqual(len(splits), 3)
        self.assertEqual("".join(splits), text)

    def test_preserves_leading_whitespace(self):
        text = "   First sentence. Second sentence."

        splits = split_sentences(text)

        self.assertEqual("".join(splits), text)

    def test_empty_text_returns_no_sentences(self):
        self.assertEqual(split_sentences(""), [])


class TestTextSplitter(unittest.TestCase):
    def test_rejects_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            TextSplitter(chunk_size=0)

    def test_short_text_remains_one_chunk(self):
        text = "This text is short."
        splitter = TextSplitter(chunk_size=100)

        chunks = splitter.split(text)

        self.assertEqual(chunks, [text])

    def test_empty_text_returns_no_chunks(self):
        splitter = TextSplitter(chunk_size=10)

        self.assertEqual(splitter.split(""), [])
        self.assertEqual(splitter.split("   \n\n"), [])

    def test_long_text_respects_chunk_size(self):
        text = (
            "This is the first paragraph with several useful words.\n\n"
            "This is the second paragraph with more useful words.\n\n"
            "This is the third paragraph and it is also fairly long."
        )
        splitter = TextSplitter(chunk_size=12)

        chunks = splitter.split(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(token_size(chunk) <= 12 for chunk in chunks)
        )

    def test_unbroken_text_uses_token_fallback(self):
        text = "a" * 1000
        splitter = TextSplitter(chunk_size=5)

        chunks = splitter.split(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(token_size(chunk) <= 5 for chunk in chunks)
        )

    def test_chunks_are_not_empty(self):
        text = "First paragraph.\n\n\n\nSecond paragraph."
        splitter = TextSplitter(chunk_size=5)

        chunks = splitter.split(text)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_instance_is_callable(self):
        text = "A short piece of text."
        splitter = TextSplitter(chunk_size=20)

        self.assertEqual(splitter(text), splitter.split(text))


if __name__ == "__main__":
    unittest.main()