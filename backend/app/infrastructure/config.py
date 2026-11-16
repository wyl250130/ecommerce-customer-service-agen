"""Settings read by the HTTP layer; scope stays server controlled."""

from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data/processed/customer_service/customer_service_train.sqlite"


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str | None
    llma_model: str
    llmb_model: str
    default_snapshot_id: str
    enable_manual_search: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_base_url=os.getenv("LLM_BASE_URL", "").strip() or None,
            llma_model=os.getenv("LLMA_MODEL", "").strip(),
            llmb_model=os.getenv("LLMB_MODEL", "").strip(),
            default_snapshot_id=os.getenv("DEFAULT_SNAPSHOT_ID", "").strip(),
            enable_manual_search=os.getenv("ENABLE_MANUAL_SEARCH", "false").lower() in {"1", "true", "yes"},
        )

    def validate_chat(self) -> None:
        missing = [name for name, value in {
            "LLM_API_KEY": self.llm_api_key,
            "LLMA_MODEL": self.llma_model,
            "LLMB_MODEL": self.llmb_model,
            "DEFAULT_SNAPSHOT_ID": self.default_snapshot_id,
        }.items() if not value]
        if missing:
            raise ValueError("Missing settings: " + ", ".join(missing))
