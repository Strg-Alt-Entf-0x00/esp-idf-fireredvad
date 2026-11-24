#!/usr/bin/env python3
"""FireRedVAD -> ESPM v3 Native Container Converter for ESP32-P4.

Converts FireRedVAD PyTorch checkpoints (model.pth.tar + cmvn.ark) into standard
ESPM version 3 container binaries (.espm) with 64-byte aligned tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("[FATAL] PyTorch not installed. Run: pip install torch")
    sys.exit(1)

try:
    import kaldiio
except ImportError:
    print("[FATAL] kaldiio not installed. Run: pip install kaldiio")
    sys.exit(1)

import numpy as np

# Import ESPM v3 container writer
from espm_format import (
    DTYPE_FP32,
    DTYPE_INT8,
    MODEL_QUANT_FP32,
    MODEL_QUANT_INT8,
    MODEL_QUANT_INT8_PER_CH,
    QUANT_NONE,
    QUANT_SYMMETRIC,
    QUANT_SYMMETRIC_AXIS,
    write_fireredvad_espm_v3,
)


def load_cmvn(cmvn_path: str) -> tuple[int, np.ndarray, np.ndarray]:
    """Load Kaldi CMVN stats and compute means + inverse standard deviation."""
    stats = kaldiio.load_mat(cmvn_path)
    assert stats.shape[0] == 2, f"CMVN must have 2 rows, got {stats.shape[0]}"
    dim = stats.shape[1] - 1
    count = stats[0, dim]
    assert count >= 1, f"CMVN count must be >= 1, got {count}"

    floor = 1e-20
    means = np.zeros(dim, dtype=np.float32)
    istd = np.zeros(dim, dtype=np.float32)

    for d in range(dim):
        mean = stats[0, d] / count
        means[d] = mean
        variance = (stats[1, d] / count) - mean * mean
        if variance < floor:
            variance = floor
        istd[d] = 1.0 / math.sqrt(variance)

    return dim, means, istd


def extract_arch_params(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """Extract DFSMN architecture parameters from state_dict shapes."""
    H, D = state_dict["dfsmn.fc1.0.weight"].shape
    P = state_dict["dfsmn.fc2.0.weight"].shape[0]
    odim = state_dict["out.weight"].shape[0]

    R = 1
    i = 0
    while f"dfsmn.fsmns.{i}.fsmn.lookback_filter.weight" in state_dict:
        R += 1
        i += 1

    M = 0
    i = 0
    while f"dfsmn.dnns.{i}.weight" in state_dict:
        M += 1
        i += 3  # Linear + ReLU + Dropout

    lb_weight = state_dict["dfsmn.fsmn1.lookback_filter.weight"]
    N1 = lb_weight.shape[2]
    S1 = 1

    N2 = 0
    S2 = 0
    if "dfsmn.fsmn1.lookahead_filter.weight" in state_dict:
        la_weight = state_dict["dfsmn.fsmn1.lookahead_filter.weight"]
        N2 = la_weight.shape[2]
        S2 = 1

    return {
        "D": int(D),
        "H": int(H),
        "P": int(P),
        "odim": int(odim),
        "R": int(R),
        "M": int(M),
        "N1": int(N1),
        "S1": int(S1),
        "N2": int(N2),
        "S2": int(S2),
    }


def detect_model_type(model_dir: str) -> str:
    basename = os.path.basename(os.path.normpath(model_dir)).lower()
    if "stream" in basename:
        return "stream-vad"
    elif "aed" in basename:
        return "aed"
    return "vad"


def export_espm(
    model_dir: str,
    output_dir: str,
    quantize_int8: bool = False,
    quantize_int8_per_ch: bool = False,
) -> Path:
    model_path = os.path.join(model_dir, "model.pth.tar")
    cmvn_path = os.path.join(model_dir, "cmvn.ark")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Compute checkpoint sha256
    with open(model_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"[INFO] Loading checkpoint: {model_path}")
    package = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = package["model_state_dict"]

    arch = extract_arch_params(state_dict)
    if "args" in package:
        args = package["args"]
        if hasattr(args, "S1"):
            arch["S1"] = int(args.S1)
        if hasattr(args, "S2"):
            arch["S2"] = int(args.S2)
        if hasattr(args, "N1"):
            arch["N1"] = int(args.N1)
        if hasattr(args, "N2"):
            arch["N2"] = int(args.N2)

    model_type_str = detect_model_type(model_dir)

    cmvn_dim = 0
    cmvn_means = np.array([], dtype=np.float32)
    cmvn_istd = np.array([], dtype=np.float32)
    if os.path.exists(cmvn_path):
        cmvn_dim, cmvn_means, cmvn_istd = load_cmvn(cmvn_path)
    arch["cmvn_dim"] = cmvn_dim

    # Determine quant type
    if quantize_int8_per_ch:
        quant_type = MODEL_QUANT_INT8_PER_CH
        quant_suffix = "int8_ch"
    elif quantize_int8:
        quant_type = MODEL_QUANT_INT8
        quant_suffix = "int8"
    else:
        quant_type = MODEL_QUANT_FP32
        quant_suffix = "fp32"

    tensors = []

    # 1. CMVN Tensors
    if cmvn_dim > 0:
        tensors.append({
            "name": "cmvn.means",
            "shape": (cmvn_dim,),
            "dtype": DTYPE_FP32,
            "quant_type": QUANT_NONE,
            "data": cmvn_means.astype("<f4").tobytes(),
        })
        tensors.append({
            "name": "cmvn.istd",
            "shape": (cmvn_dim,),
            "dtype": DTYPE_FP32,
            "quant_type": QUANT_NONE,
            "data": cmvn_istd.astype("<f4").tobytes(),
        })

    def process_tensor(canonical_name: str, tensor: torch.Tensor, is_bias: bool = False, is_filter: bool = False):
        arr = tensor.detach().cpu().numpy().astype(np.float32)

        if is_filter:
            # Squeeze channel dim [P, 1, N] -> [P, N], time-reverse, transpose to [N, P]
            arr = arr.squeeze(1)[:, ::-1].transpose(1, 0)
            arr = np.ascontiguousarray(arr)

        shape = arr.shape

        if is_bias or quant_type == MODEL_QUANT_FP32:
            return {
                "name": canonical_name,
                "shape": shape,
                "dtype": DTYPE_FP32,
                "quant_type": QUANT_NONE,
                "data": arr.astype("<f4").tobytes(),
            }

        if quant_type == MODEL_QUANT_INT8_PER_CH:
            if is_filter:
                # Per-tensor for depthwise FSMN filters
                max_abs = float(np.max(np.abs(arr)))
                scale = max_abs / 127.0 if max_abs > 0 else 1.0
                q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
                return {
                    "name": canonical_name,
                    "shape": shape,
                    "dtype": DTYPE_INT8,
                    "quant_type": QUANT_SYMMETRIC,
                    "data": q.tobytes(),
                    "scales": np.array([scale], dtype="<f4").tobytes(),
                    "num_channels": 1,
                    "channel_axis": 255,
                }
            elif len(shape) > 1:
                # Per-channel for weight matrices [out_channels, in_channels]
                num_channels = shape[0]
                scales = np.zeros(num_channels, dtype=np.float32)
                q = np.zeros_like(arr, dtype=np.int8)
                for c in range(num_channels):
                    row = arr[c]
                    max_abs = float(np.max(np.abs(row)))
                    scale = max_abs / 127.0 if max_abs > 0 else 1.0
                    scales[c] = scale
                    q[c] = np.clip(np.round(row / scale), -127, 127).astype(np.int8)
                return {
                    "name": canonical_name,
                    "shape": shape,
                    "dtype": DTYPE_INT8,
                    "quant_type": QUANT_SYMMETRIC_AXIS,
                    "data": q.tobytes(),
                    "scales": scales.astype("<f4").tobytes(),
                    "num_channels": num_channels,
                    "channel_axis": 0,
                }
            else:
                return {
                    "name": canonical_name,
                    "shape": shape,
                    "dtype": DTYPE_FP32,
                    "quant_type": QUANT_NONE,
                    "data": arr.astype("<f4").tobytes(),
                }

        elif quant_type == MODEL_QUANT_INT8:
            max_abs = float(np.max(np.abs(arr)))
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
            return {
                "name": canonical_name,
                "shape": shape,
                "dtype": DTYPE_INT8,
                "quant_type": QUANT_SYMMETRIC,
                "data": q.tobytes(),
                "scales": np.array([scale], dtype="<f4").tobytes(),
                "num_channels": 1,
                "channel_axis": 255,
            }

    # 2. FC1 and FC2
    tensors.append(process_tensor("fc1.weight", state_dict["dfsmn.fc1.0.weight"]))
    tensors.append(process_tensor("fc1.bias", state_dict["dfsmn.fc1.0.bias"], is_bias=True))
    tensors.append(process_tensor("fc2.weight", state_dict["dfsmn.fc2.0.weight"]))
    tensors.append(process_tensor("fc2.bias", state_dict["dfsmn.fc2.0.bias"], is_bias=True))

    # 3. FSMN 1
    tensors.append(process_tensor("fsmn1.lookback_weight", state_dict["dfsmn.fsmn1.lookback_filter.weight"], is_filter=True))
    if "dfsmn.fsmn1.lookahead_filter.weight" in state_dict:
        tensors.append(process_tensor("fsmn1.lookahead_weight", state_dict["dfsmn.fsmn1.lookahead_filter.weight"], is_filter=True))

    # 4. R-1 DFSMN Blocks
    for b in range(arch["R"] - 1):
        prefix = f"dfsmn.fsmns.{b}"
        tensors.append(process_tensor(f"block_fc1.{b}.weight", state_dict[f"{prefix}.fc1.0.weight"]))
        tensors.append(process_tensor(f"block_fc1.{b}.bias", state_dict[f"{prefix}.fc1.0.bias"], is_bias=True))
        tensors.append(process_tensor(f"block_fc2.{b}.weight", state_dict[f"{prefix}.fc2.weight"]))
        tensors.append(process_tensor(f"block_fsmn.{b}.lookback_weight", state_dict[f"{prefix}.fsmn.lookback_filter.weight"], is_filter=True))
        if f"{prefix}.fsmn.lookahead_filter.weight" in state_dict:
            tensors.append(process_tensor(f"block_fsmn.{b}.lookahead_weight", state_dict[f"{prefix}.fsmn.lookahead_filter.weight"], is_filter=True))

    # 5. DNN layers
    for d in range(arch["M"]):
        idx = d * 3
        tensors.append(process_tensor(f"dnn.{d}.weight", state_dict[f"dfsmn.dnns.{idx}.weight"]))
        tensors.append(process_tensor(f"dnn.{d}.bias", state_dict[f"dfsmn.dnns.{idx}.bias"], is_bias=True))

    # 6. Output layer
    tensors.append(process_tensor("out.weight", state_dict["out.weight"]))
    tensors.append(process_tensor("out.bias", state_dict["out.bias"], is_bias=True))

    out_name = f"firered_{model_type_str}_{quant_suffix}.espm"
    out_path = Path(output_dir) / out_name

    res = write_fireredvad_espm_v3(
        out_path,
        model_name=f"firered_{model_type_str}",
        version_str="1.0.0",
        model_type_str=model_type_str,
        arch=arch,
        quant_type=quant_type,
        tensors=tensors,
        checkpoint_sha256=ckpt_sha256,
    )

    print(f"[OK] Wrote ESPM v3: {out_path} ({res['file_size']:,} bytes, {res['num_tensors']} tensors, PSRAM: {res['psram_required_kb']} KB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="FireRedVAD -> ESPM v3 Native Container Converter")
    parser.add_argument("--model-dir", help="Path to single model directory (with model.pth.tar and cmvn.ark)")
    parser.add_argument("--output-dir", required=True, help="Output directory for .espm models")
    parser.add_argument("--all", action="store_true", help="Convert all models in models_py/")
    parser.add_argument("--quantize-int8", action="store_true", help="INT8 symmetric per-tensor quantization")
    parser.add_argument("--quantize-int8-per-ch", action="store_true", help="INT8 symmetric per-channel quantization")

    args = parser.parse_args()

    if args.all:
        script_dir = Path(__file__).resolve().parent
        models_root = script_dir.parents[1] / "models" / "py"
        if not models_root.exists():
            models_root = script_dir.parents[1] / "models_py"
        if not models_root.exists():
            print(f"[FATAL] Model root not found at {models_root}")
            sys.exit(1)

        count = 0
        for entry in sorted(os.listdir(models_root)):
            mdir = models_root / entry
            if mdir.is_dir() and (mdir / "model.pth.tar").exists():
                print(f"\n--- Converting {entry} ---")
                export_espm(str(mdir), args.output_dir, args.quantize_int8, args.quantize_int8_per_ch)
                count += 1
        print(f"\n[OK] Converted {count} model(s) to ESPM v3.")
    elif args.model_dir:
        export_espm(args.model_dir, args.output_dir, args.quantize_int8, args.quantize_int8_per_ch)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
