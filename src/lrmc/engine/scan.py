"""Universal "scan" path for RAW EXECUTABLES (any file).

Reads bytes only via :class:`~lrmc.data.preprocessor.FilePreprocessor`, NEVER
executes or parses the file as an executable container beyond an optional
fail-soft PE-header sanity check, enforces the size limit, computes sha256,
renders the same image representation used in training, and runs it through
the frozen inference engine -> Alert Generator.
"""

from __future__ import annotations

from pathlib import Path

from lrmc.data.image_generator import ImageGenerator
from lrmc.data.preprocessor import FilePreprocessor
from lrmc.engine.alert import Alert, AlertGenerator
from lrmc.engine.infer import FrozenInferenceEngine


class Scanner:
    def __init__(
        self,
        engine: FrozenInferenceEngine,
        image_gen: ImageGenerator,
        alert_gen: AlertGenerator,
    ):
        self.engine = engine
        self.image_gen = image_gen
        self.alert_gen = alert_gen
        self.preprocessor = FilePreprocessor()

    def scan_file(self, path: str | Path) -> Alert:
        pre = self.preprocessor.load_raw_executable(path)
        pe_check = self.preprocessor.pe_header_sanity_check(pre.byte_stream)  # fail-soft, informational
        image = self.image_gen.eval_view(pre.byte_stream, is_pre_rendered_image=False)
        result = self.engine.predict(image)
        alert = self.alert_gen.make_alert(
            pre.sha256,
            result,
            self.engine.model_hash,
            source_path=str(path),
            pe_header=pe_check,
        )
        self.alert_gen.emit(alert)
        return alert

    def scan_many(self, paths: list[str | Path]) -> list[Alert]:
        return [self.scan_file(p) for p in paths]
