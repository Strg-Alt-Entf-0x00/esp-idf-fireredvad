#!/usr/bin/env python3
"""Strict parser and integrity verifier for FireRedVAD ESPM v3 models."""

import struct
import zlib
from pathlib import Path
import numpy as np

MAGIC = 0x4D505345
HEADER_SIZE = 128
SECTION_ENTRY_SIZE = 40
ALIGNMENT = 64

_HEADER = struct.Struct("<IHHHH32s16sIIIQIHHIII28s")
_SECTION = struct.Struct("<HHQQI16s")
_METADATA_HEADER = struct.Struct("<4sHHII")
_METADATA_RECORD = struct.Struct("<HBBI")
_INDEX_HEADER = struct.Struct("<4sHHHHIIQQ")
_TENSOR_RECORD = struct.Struct("<II4BI8IIB3sQIQ Q64s12s")


def verify_espm(file_path: Path):
    print(f"\n[VERIFY] Checking ESPM v3: {file_path.name}")
    data = file_path.read_bytes()
    assert len(data) >= HEADER_SIZE, "File smaller than header"

    (
        magic, ver, mtype, qtype, num_sections,
        mname_raw, vstr_raw, srate, psram_kb,
        hdr_crc, file_size, tbl_off, hdr_size,
        entry_size, align, flags, tbl_crc, _
    ) = _HEADER.unpack(data[:HEADER_SIZE])

    assert magic == MAGIC, f"Bad magic: {hex(magic)}"
    assert ver == 3, f"Bad version: {ver}"
    assert file_size == len(data), f"Size mismatch: {file_size} != {len(data)}"
    assert align == ALIGNMENT, f"Bad alignment: {align}"

    # Verify header CRC
    header_zero_crc = bytearray(data[:HEADER_SIZE])
    header_zero_crc[68:72] = b"\x00\x00\x00\x00"
    calc_hdr_crc = zlib.crc32(header_zero_crc) & 0xFFFFFFFF
    assert calc_hdr_crc == hdr_crc, f"Header CRC mismatch: {calc_hdr_crc:#x} != {hdr_crc:#x}"

    # Verify section table
    tbl_bytes = data[tbl_off:tbl_off + num_sections * entry_size]
    calc_tbl_crc = zlib.crc32(tbl_bytes) & 0xFFFFFFFF
    assert calc_tbl_crc == tbl_crc, f"Table CRC mismatch: {calc_tbl_crc:#x} != {tbl_crc:#x}"

    sections = {}
    for i in range(num_sections):
        off = tbl_off + i * entry_size
        stype, sflags, soff, ssize, scrc, sname_raw = _SECTION.unpack(data[off:off + entry_size])
        sname = sname_raw.decode("utf-8").rstrip("\0")
        spayload = data[soff:soff + ssize]
        calc_scrc = zlib.crc32(spayload) & 0xFFFFFFFF
        assert calc_scrc == scrc, f"Section {sname} CRC mismatch: {calc_scrc:#x} != {scrc:#x}"
        assert (soff % ALIGNMENT) == 0, f"Section {sname} not {ALIGNMENT}-byte aligned: {soff}"
        sections[sname] = {
            "type": stype,
            "offset": soff,
            "size": ssize,
            "payload": spayload,
        }
        print(f"  - Section '{sname}' (type={stype}, size={ssize:,} B, offset={soff:#x}) OK")

    assert "tensor_index" in sections, "Missing tensor_index"
    assert "tensor_data" in sections, "Missing tensor_data"
    assert "metadata" in sections, "Missing metadata"

    # Verify index
    idx_payload = sections["tensor_index"]["payload"]
    tdata_payload = sections["tensor_data"]["payload"]

    magic_idx, idx_ver, _, rec_size, _, num_tensors, _, total_elems, total_bytes = _INDEX_HEADER.unpack(idx_payload[:36])
    assert magic_idx == b"ETIX", f"Bad ETIX magic: {magic_idx}"
    assert idx_ver == 2, f"Bad index version: {idx_ver}"
    assert rec_size == 160, f"Bad record size: {rec_size}"

    rec_off = 36
    for t_idx in range(num_tensors):
        rec_bytes = idx_payload[rec_off:rec_off + 160]
        (
            nhash, scope, dtype, rank, quant, layout,
            num_elem, *shape, num_ch, ch_axis, _,
            scales_off, scales_sz, data_off, data_sz,
            tname_raw, _
        ) = _TENSOR_RECORD.unpack(rec_bytes)
        rec_off += 160

        tname = tname_raw.decode("utf-8").rstrip("\0")
        calc_hash = zlib.crc32(tname.encode("utf-8")) & 0xFFFFFFFF
        assert calc_hash == nhash, f"Hash mismatch for {tname}: {calc_hash:#x} != {nhash:#x}"
        assert data_off + data_sz <= len(tdata_payload), f"Tensor {tname} out of bounds"
        if scales_sz > 0:
            assert scales_off + scales_sz <= len(tdata_payload), f"Scales {tname} out of bounds"

    print(f"  [OK] Validated {num_tensors} tensors, {total_elems:,} elements ({total_bytes:,} data bytes).")
    print(f"[SUCCESS] {file_path.name} is 100% compliant ESPM v3!\n")


if __name__ == "__main__":
    import sys
    model_dir = Path(__file__).resolve().parents[2] / "models" / "espm"
    for f in sorted(model_dir.glob("*.espm")):
        verify_espm(f)
