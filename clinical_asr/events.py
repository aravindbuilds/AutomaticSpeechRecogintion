from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TranscriptEvent:
    type: str
    text: str = ""
    confidence: float | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    terms: list[dict[str, Any]] | None = None
    session_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}
