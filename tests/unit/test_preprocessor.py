import numpy as np
import pytest

from lrmc.data.preprocessor import (
    MAX_SCAN_FILE_BYTES,
    FilePreprocessor,
)


def test_bytes_file_parsing_basic(tmp_path):
    p = tmp_path / "sample.bytes"
    p.write_text("00401000 4D 5A 90 00\n00401004 03 00 00 00\n")
    pre = FilePreprocessor()
    result = pre.load_bytes_file(p)
    assert result.byte_stream.tolist() == [0x4D, 0x5A, 0x90, 0x00, 0x03, 0x00, 0x00, 0x00]
    assert result.unknown_byte_frac == 0.0
    assert result.source_kind == "bytes_file"
    assert len(result.sha256) == 64


def test_bytes_file_handles_unknown_tokens(tmp_path):
    p = tmp_path / "sanitized.bytes"
    p.write_text("00401000 4D ?? 90 ??\n")
    pre = FilePreprocessor()
    result = pre.load_bytes_file(p)
    assert result.byte_stream.tolist() == [0x4D, 0, 0x90, 0]
    assert result.unknown_byte_frac == 0.5


def test_bytes_file_empty_lines_ignored(tmp_path):
    p = tmp_path / "sample.bytes"
    p.write_text("00401000 4D 5A\n\n\n00401002 90 00\n")
    pre = FilePreprocessor()
    result = pre.load_bytes_file(p)
    assert result.byte_stream.size == 4


def test_raw_executable_reads_bytes_only(tmp_path):
    p = tmp_path / "sample.exe"
    payload = bytes(range(256)) * 4
    p.write_bytes(payload)
    pre = FilePreprocessor()
    result = pre.load_raw_executable(p)
    assert result.source_kind == "raw_executable"
    assert result.byte_stream.size == len(payload)
    assert result.byte_stream.tobytes() == payload


def test_raw_executable_size_limit_enforced(tmp_path):
    p = tmp_path / "huge.bin"
    # sparse-ish large file without actually allocating MAX_SCAN_FILE_BYTES in memory
    with open(p, "wb") as fh:
        fh.seek(MAX_SCAN_FILE_BYTES + 1)
        fh.write(b"\x00")
    pre = FilePreprocessor()
    with pytest.raises(ValueError):
        pre.load_raw_executable(p)


def test_pe_header_sanity_check_fails_soft_on_garbage():
    pre = FilePreprocessor()
    garbage = np.random.randint(0, 256, size=10, dtype=np.uint8)
    result = pre.pe_header_sanity_check(garbage)
    assert result["has_mz"] is False
    assert result["has_pe"] is False


def test_pe_header_sanity_check_detects_mz_pe():
    pre = FilePreprocessor()
    buf = bytearray(256)
    buf[0:2] = b"MZ"
    buf[0x3C] = 0x80
    buf[0x80:0x84] = b"PE\x00\x00"
    arr = np.frombuffer(bytes(buf), dtype=np.uint8)
    result = pre.pe_header_sanity_check(arr)
    assert result["has_mz"] is True
    assert result["has_pe"] is True


def test_load_image_array_grayscale(tmp_path):
    from PIL import Image

    arr = (np.random.rand(40, 30) * 255).astype(np.uint8)
    img_path = tmp_path / "img.png"
    Image.fromarray(arr).convert("L").save(img_path)
    pre = FilePreprocessor()
    result = pre.load_image_array(img_path)
    assert result.byte_stream.shape == (40, 30)
    assert result.source_kind == "image"
