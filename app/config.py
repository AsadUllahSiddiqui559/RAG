from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    groq_api_key: SecretStr
    groq_model: str = "openai/gpt-oss-120b"

    nomic_model: str = "nomic-embed-text-v1.5"
    nomic_inference_mode: Literal["local", "remote"] = "local"

    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    vector_store_path: Path = PROJECT_ROOT / "data" / "vector_store.json"

    chunk_size: int = Field(default=512, gt=0)
    top_k: int = Field(default=5, gt=0)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("groq_api_key")
    @classmethod
    def validate_groq_api_key(cls, value: SecretStr) -> SecretStr:
        api_key = value.get_secret_value().strip()

        invalid_values = {
            "",
            "replace_with_your_groq_api_key",
            "your_actual_key",
        }

        if api_key in invalid_values:
            raise ValueError("GROQ_API_KEY has not been configured")

        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()