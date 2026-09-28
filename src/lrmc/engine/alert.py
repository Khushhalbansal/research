"""Alert Generator (diagram box: "Alert Generator").

Emits one JSONL record per scanned/predicted sample: sha256, verdict, nearest
family, per-class distance/radius ratios, score, timestamp, model hash.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lrmc.engine.infer import InferenceOutput


@dataclass
class Alert:
    sha256: str
    verdict: str
    nearest_family: str
    score: float
    distances: dict[str, float]
    ratios: dict[str, float]
    timestamp: str
    model_hash: str
    source_path: str = ""
    pe_header: dict | None = None

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class AlertGenerator:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def make_alert(
        self,
        sha256: str,
        result: InferenceOutput,
        model_hash: str,
        source_path: str = "",
        pe_header: dict | None = None,
    ) -> Alert:
        return Alert(
            sha256=sha256,
            verdict=result.verdict,
            nearest_family=result.nearest_family,
            score=result.score,
            distances=result.distances,
            ratios=result.ratios,
            timestamp=datetime.now(UTC).isoformat(),
            model_hash=model_hash,
            source_path=source_path,
            pe_header=pe_header,
        )

    def emit(self, alert: Alert) -> None:
        with open(self.output_path, "a", encoding="utf-8") as fh:
            fh.write(alert.to_json_line() + "\n")

    def emit_all(self, alerts: list[Alert]) -> None:
        with open(self.output_path, "a", encoding="utf-8") as fh:
            for a in alerts:
                fh.write(a.to_json_line() + "\n")
