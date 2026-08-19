import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings


class TestSettings(unittest.TestCase):
    def test_default_configuration(self):
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key_for_configuration"},
            clear=True,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.groq_model, "openai/gpt-oss-120b")
        self.assertEqual(settings.nomic_model, "nomic-embed-text-v1.5")
        self.assertEqual(settings.nomic_inference_mode, "local")
        self.assertEqual(settings.chunk_size, 512)
        self.assertEqual(settings.top_k, 5)
        self.assertEqual(settings.docs_dir, PROJECT_ROOT / "data" / "docs")
        self.assertEqual(
            settings.vector_store_path,
            PROJECT_ROOT / "data" / "vector_store.json",
        )

    def test_missing_api_key_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)

    def test_placeholder_api_key_is_rejected(self):
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "replace_with_your_groq_api_key"},
            clear=True,
        ):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)

    def test_invalid_nomic_mode_is_rejected(self):
        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "test_key_for_configuration",
                "NOMIC_INFERENCE_MODE": "invalid",
            },
            clear=True,
        ):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)

    def test_invalid_numeric_values_are_rejected(self):
        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "test_key_for_configuration",
                "CHUNK_SIZE": "0",
                "TOP_K": "-1",
            },
            clear=True,
        ):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)


if __name__ == "__main__":
    unittest.main()