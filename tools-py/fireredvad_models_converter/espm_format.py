"""FireRedVAD ESPM v3 container writer and validator."""

from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

MAGIC = 0x4D505345  # "ESPM" in little-endian
HEADER_SIZE = 128
SECTION_ENTRY_SIZE = 40
ALIGNMENT = 64
V3_FORMAT_VERSION = 3
V3_MAX_SECTIONS = 16
V3_INDEX_HEADER_SIZE = 36
V3_TENSOR_RECORD_SIZE = 160
V3_MAX_TENSORS = 512
V3_SECTION_OPTIONAL = 1

V3_METADATA_HEADER = struct.Struct("<4sHHII")
V3_METADATA_RECORD = struct.Struct("<HBBI")
V3_INDEX_HEADER = struct.Struct("<4sHHHHIIQQ")
V3_TENSOR = struct.Struct("<II4BI8IIB3sQIQ Q64s12s")

assert V3_METADATA_HEADER.size == 16
assert V3_METADATA_RECORD.size == 8
assert V3_INDEX_HEADER.size == V3_INDEX_HEADER_SIZE
assert V3_TENSOR.size == V3_TENSOR_RECORD_SIZE

MODEL_TYPE_VAD = 1
MODEL_QUANT_NONE = 0
MODEL_QUANT_INT8 = 1
MODEL_QUANT_INT8_PER_CH = 2
MODEL_QUANT_FP32 = 4

SECTION_CONFIG_JSON = 5
SECTION_TENSOR_INDEX = 6
SECTION_TENSOR_DATA = 7
SECTION_METADATA = 8

DTYPE_INT8 = 1
DTYPE_FP32 = 2

QUANT_NONE = 0
QUANT_SYMMETRIC = 1
QUANT_SYMMETRIC_AXIS = 2

_HEADER = struct.Struct("<IHHHH32s16sIIIQIHHIII28s")
_SECTION = struct.Struct("<HHQQI16s")
assert _HEADER.size == HEADER_SIZE
assert _SECTION.size == SECTION_ENTRY_SIZE


def _align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _fixed_string(value: str, capacity: int, field: str) -> bytes:
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) >= capacity or b"\0" in encoded:
        raise ValueError(f"{field} must fit in {capacity - 1} UTF-8 bytes without internal NUL")
    return encoded + bytes(capacity - len(encoded))


def _encode_v3_metadata(records: dict[str, tuple[int, bytes]]) -> bytes:
    encoded = bytearray(V3_METADATA_HEADER.pack(b"EMET", 1, 16, len(records), 0))
    for key in sorted(records):
        key_bytes = key.encode("utf-8")
        value_type, value = records[key]
        if not key_bytes or b"\0" in key_bytes or len(key_bytes) > 0xFFFF:
            raise ValueError(f"invalid metadata key {key!r}")
        encoded.extend(V3_METADATA_RECORD.pack(len(key_bytes), value_type, 1, len(value)))
        encoded.extend(key_bytes)
        encoded.extend(value)
    if len(encoded) > 64 * 1024:
        raise ValueError("typed metadata exceeds 64 KiB")
    return bytes(encoded)


def _meta_u32(value: int) -> tuple[int, bytes]:
    return 1, struct.pack("<I", int(value) & 0xFFFFFFFF)


def _meta_f32(value: float) -> tuple[int, bytes]:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("metadata float must be finite")
    return 4, struct.pack("<f", value)


def _meta_utf8(value: str) -> tuple[int, bytes]:
    encoded = value.encode("utf-8")
    if b"\0" in encoded:
        raise ValueError("metadata string cannot contain NUL")
    return 6, encoded


def write_fireredvad_espm_v3(
    output_path: Path,
    *,
    model_name: str,
    version_str: str,
    model_type_str: str,
    arch: dict[str, int],
    quant_type: int,
    tensors: list[dict[str, Any]],
    checkpoint_sha256: str = "",
) -> dict[str, Any]:
    """Writes a standard ESPM v3 container for FireRedVAD DFSMN.
    
    Each tensor dict contains:
      - name: str
      - shape: tuple[int, ...]
      - dtype: int (DTYPE_FP32 or DTYPE_INT8)
      - quant_type: int (QUANT_NONE, QUANT_SYMMETRIC, QUANT_SYMMETRIC_AXIS)
      - data: bytes
      - scales: bytes (optional)
      - num_channels: int (1 for per-tensor, out_channels for per-channel)
      - channel_axis: int (255 for per-tensor, 0 for per-channel)
    """
    tensor_data = bytearray()
    tensor_records = []
    tensor_names = set()
    total_elements = 0
    total_data_bytes = 0

    for t in tensors:
        name = t["name"]
        if name in tensor_names:
            raise ValueError(f"duplicate tensor name: {name}")
        tensor_names.add(name)

        shape = tuple(t["shape"])
        dtype = t.get("dtype", DTYPE_FP32)
        t_quant = t.get("quant_type", QUANT_NONE)
        raw_data = t["data"]
        raw_scales = t.get("scales", b"")
        num_channels = t.get("num_channels", 1)
        channel_axis = t.get("channel_axis", 255)

        elements = math.prod(shape) if shape else 1
        
        # 64-byte align each tensor payload in tensor_data
        pad_needed = (-len(tensor_data)) % ALIGNMENT
        tensor_data.extend(bytes(pad_needed))
        
        scales_offset = 0
        scales_size = 0
        if raw_scales:
            scales_offset = len(tensor_data)
            scales_size = len(raw_scales)
            tensor_data.extend(raw_scales)
            pad_scales = (-len(tensor_data)) % 16
            tensor_data.extend(bytes(pad_scales))

        data_offset = len(tensor_data)
        data_size = len(raw_data)
        tensor_data.extend(raw_data)

        name_bytes = _fixed_string(name, 64, "tensor name")
        name_hash = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
        dimensions = shape + (0,) * (8 - len(shape))

        tensor_records.append(V3_TENSOR.pack(
            name_hash,
            0,             # scope_id
            dtype,
            len(shape),
            t_quant,
            0,             # layout: contiguous row-major
            elements,
            *dimensions,
            num_channels,
            channel_axis,
            bytes(3),      # reserved
            scales_offset,
            scales_size,
            data_offset,
            data_size,
            name_bytes,
            bytes(12),     # reserved_v2
        ))

        total_elements += elements
        total_data_bytes += data_size

    # Build Metadata records
    metadata_records = {
        "general.architecture": _meta_utf8("fireredvad.dfsmn"),
        "general.architecture_version": _meta_u32(1),
        "fireredvad.model_type": _meta_utf8(model_type_str),
        "fireredvad.D": _meta_u32(arch["D"]),
        "fireredvad.H": _meta_u32(arch["H"]),
        "fireredvad.P": _meta_u32(arch["P"]),
        "fireredvad.odim": _meta_u32(arch["odim"]),
        "fireredvad.R": _meta_u32(arch["R"]),
        "fireredvad.M": _meta_u32(arch["M"]),
        "fireredvad.N1": _meta_u32(arch["N1"]),
        "fireredvad.S1": _meta_u32(arch["S1"]),
        "fireredvad.N2": _meta_u32(arch["N2"]),
        "fireredvad.S2": _meta_u32(arch["S2"]),
        "fireredvad.cmvn_dim": _meta_u32(arch.get("cmvn_dim", 80)),
    }
    if checkpoint_sha256:
        metadata_records["fireredvad.checkpoint_sha256"] = _meta_utf8(checkpoint_sha256.lower())

    metadata_bytes = _encode_v3_metadata(metadata_records)

    # Config JSON
    config_dict = {
        "architecture": "fireredvad.dfsmn",
        "model_name": model_name,
        "version": version_str,
        "model_type": model_type_str,
        "sample_rate": 16000,
        "arch": arch,
        "num_tensors": len(tensors),
        "total_elements": total_elements,
    }
    config_json_bytes = json.dumps(config_dict, indent=2).encode("utf-8")

    # ETIX index
    index_header = V3_INDEX_HEADER.pack(
        b"ETIX",
        2,  # version 2
        V3_INDEX_HEADER_SIZE,
        V3_TENSOR_RECORD_SIZE,
        0,  # flags
        len(tensor_records),
        1,  # scope_count
        total_elements,
        total_data_bytes,
    )
    index_bytes = index_header + b"".join(tensor_records)

    sections = [
        (SECTION_CONFIG_JSON, 0, "config_json", config_json_bytes),
        (SECTION_TENSOR_INDEX, 0, "tensor_index", index_bytes),
        (SECTION_TENSOR_DATA, 0, "tensor_data", bytes(tensor_data)),
        (SECTION_METADATA, 0, "metadata", metadata_bytes),
    ]

    table_end = HEADER_SIZE + len(sections) * SECTION_ENTRY_SIZE
    offset = _align(table_end, ALIGNMENT)
    rows = []
    for kind, section_flags, name, payload in sections:
        rows.append((kind, section_flags, name, offset, payload, zlib.crc32(payload) & 0xFFFFFFFF))
        offset = _align(offset + len(payload), ALIGNMENT)

    file_size = rows[-1][3] + len(rows[-1][4])
    section_table = b"".join(_SECTION.pack(
        kind, section_flags, section_offset, len(payload), checksum,
        _fixed_string(name, 16, "section name")
    ) for kind, section_flags, name, section_offset, payload, checksum in rows)

    table_crc = zlib.crc32(section_table) & 0xFFFFFFFF
    psram_required_kb = (len(tensor_data) + 1023) // 1024

    def pack_header(header_crc: int) -> bytes:
        return _HEADER.pack(
            MAGIC,
            V3_FORMAT_VERSION,
            MODEL_TYPE_VAD,
            quant_type,
            len(rows),
            _fixed_string(model_name, 32, "model name"),
            _fixed_string(version_str, 16, "version"),
            16000,              # sample_rate
            psram_required_kb,
            header_crc,
            file_size,
            HEADER_SIZE,
            HEADER_SIZE,
            SECTION_ENTRY_SIZE,
            ALIGNMENT,
            0,                  # flags
            table_crc,
            bytes(28),          # reserved
        )

    hdr = pack_header(0)
    final_header = pack_header(zlib.crc32(hdr) & 0xFFFFFFFF)

    output = bytearray(file_size)
    output[:HEADER_SIZE] = final_header
    output[HEADER_SIZE:HEADER_SIZE + len(section_table)] = section_table
    for _, _, _, s_offset, payload, _ in rows:
        output[s_offset:s_offset + len(payload)] = payload

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)

    return {
        "output_path": str(output_path),
        "file_size": file_size,
        "num_tensors": len(tensors),
        "total_elements": total_elements,
        "psram_required_kb": psram_required_kb,
        "table_crc": table_crc,
    }
