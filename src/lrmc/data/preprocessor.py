"""File Preprocessor (diagram box: "File Preprocessor").

Turns a file on disk into a raw byte stream (``np.ndarray[uint8]``), the shared
input contract for :class:`lrmc.data.image_generator.ImageGenerator`. Three source
kinds are supported:

* BIG 2015 ``.bytes`` files -- hex-dump text with an address prefix per line and
  occasional ``??`` tokens for sanitized/unknown bytes.
* Pre-rendered images (Malimg PNGs) -- read as a byte stream is meaningless here;
  :func:`load_image_array` reads pixels directly instead.
* Raw, arbitrary executables via the "scan" path -- :func:`load_raw_executable`
  reads bytes only, NEVER parses the file as a PE/ELF/etc. or executes it, and
  enforces a hard size limit so a hostile huge file can't exhaust memory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# `??` denotes a sanitized/unknown byte in the Microsoft BIG 2015 release (headers
# and some regions were scrubbed for the competition). We map it to 0, matching the
# padding/background value used elsewhere in the pipeline, and separately report the
# fraction of unknown bytes per file so downstream code/paper text can quantify how
# much of a sample was sanitized rather than silently treating it as ordinary data.
UNKNOWN_BYTE_TOKEN = "??"
UNKNOWN_BYTE_FILL = 0

# Hard ceiling for the raw-executable scan path: refuse to read arbitrarily large
# files into memory. 64 MiB comfortably covers the vast majority of real-world PE
# files while bounding worst-case memory use for a service that scans untrusted input.
MAX_SCAN_FILE_BYTES = 64 * 1024 * 1024


@dataclass
class PreprocessedFile:
    byte_stream: np.ndarray  # 1D uint8
    sha256: str
    source_kind: str  # "bytes_file" | "image" | "raw_executable"
    unknown_byte_frac: float = 0.0
    original_size: int = 0


class FilePreprocessor:
    """Loads a file into a raw byte stream, format-aware but never format-executing."""

    def load_bytes_file(self, path: str | Path) -> PreprocessedFile:
        """Parse a BIG-2015-style ``.bytes`` hex dump into a byte stream."""
        path = Path(path)
        values: list[int] = []
        n_unknown = 0
        n_total = 0
        with open(path, encoding="ascii", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if not tokens:
                    continue
                # first token is the address prefix; skip it
                for tok in tokens[1:]:
                    n_total += 1
                    if tok == UNKNOWN_BYTE_TOKEN:
                        values.append(UNKNOWN_BYTE_FILL)
                        n_unknown += 1
                    else:
                        try:
                            values.append(int(tok, 16))
                        except ValueError:
                            # Any other malformed token is treated like an
                            # unknown byte rather than crashing the pipeline.
                            values.append(UNKNOWN_BYTE_FILL)
                            n_unknown += 1
        arr = np.asarray(values, dtype=np.uint8)
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        frac_unknown = (n_unknown / n_total) if n_total else 0.0
        return PreprocessedFile(
            byte_stream=arr,
            sha256=digest,
            source_kind="bytes_file",
            unknown_byte_frac=frac_unknown,
            original_size=arr.size,
        )

    def load_image_array(self, path: str | Path) -> PreprocessedFile:
        """Load a pre-rendered grayscale malware image (e.g. Malimg PNG)."""
        path = Path(path)
        with Image.open(path) as im:
            im = im.convert("L")
            arr = np.array(im, dtype=np.uint8)
        raw_bytes = Path(path).read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        # Kept 2D (not flattened): image_generator needs the true width/height,
        # since a pre-rendered image is not re-derived from the Nataraj table.
        return PreprocessedFile(
            byte_stream=arr,
            sha256=digest,
            source_kind="image",
            unknown_byte_frac=0.0,
            original_size=arr.size,
        )

    def load_raw_executable(self, path: str | Path) -> PreprocessedFile:
        """Read raw bytes of an arbitrary file for the inference "scan" path.

        Safety contract: this function only ever calls ``open(...).read()``. It
        does not parse the file as a PE/ELF/Mach-O container, does not invoke any
        interpreter, and does not execute or spawn the file in any way. Size is
        bounded by :data:`MAX_SCAN_FILE_BYTES` to avoid unbounded memory use on a
        hostile or corrupted input.
        """
        path = Path(path)
        size = path.stat().st_size
        if size > MAX_SCAN_FILE_BYTES:
            raise ValueError(
                f"File {path} is {size} bytes, exceeding the scan limit of "
                f"{MAX_SCAN_FILE_BYTES} bytes; refusing to load."
            )
        raw = path.read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        digest = hashlib.sha256(raw).hexdigest()
        return PreprocessedFile(
            byte_stream=arr,
            sha256=digest,
            source_kind="raw_executable",
            unknown_byte_frac=0.0,
            original_size=arr.size,
        )

    def pe_header_sanity_check(self, byte_stream: np.ndarray) -> dict:
        """Optional, fail-soft check for an 'MZ'/'PE' signature.

        Never raises: a missing or malformed header is common (BIG 2015 files
        are sanitized and have no header at all) and must not break the scan
        path. Returns a small diagnostic dict instead.
        """
        try:
            if byte_stream.size < 64:
                return {"has_mz": False, "has_pe": False, "note": "file too short"}
            has_mz = bytes(byte_stream[:2].tolist()) == b"MZ"
            has_pe = False
            if has_mz and byte_stream.size >= 0x40:
                pe_offset = int(byte_stream[0x3C]) | (int(byte_stream[0x3D]) << 8)
                if pe_offset + 4 <= byte_stream.size:
                    sig = bytes(byte_stream[pe_offset : pe_offset + 4].tolist())
                    has_pe = sig == b"PE\x00\x00"
            return {"has_mz": has_mz, "has_pe": has_pe, "note": ""}
        except Exception as exc:  # noqa: BLE001 - fail-soft by design
            return {"has_mz": False, "has_pe": False, "note": f"check failed: {exc}"}
