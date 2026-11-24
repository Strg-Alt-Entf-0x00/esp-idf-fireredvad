# Changelog

## 1.1.0 - 2026-11-24

### Added
- **WiFi 6 & HTTP Transfer Service**: Integrated `wifi_transfer.cpp` and `wifi_transfer.h` in `examples/ws-esp32-p4-wifi6-console-vad-dev` supporting ESP-Hosted SDIO WiFi 6 (ESP32-C6 slave) and HTTP file uploads for models and WAV files directly to SD card.
- **WAV Container Engine**: Integrated `audio_wav.c` / `audio_wav.h` for native RIFF/WAVE parsing and streaming.
- **Scientific Corpus Evaluation Suite**: Created `tools-py/fireredvad_validation/fireredvad_evaluate_corpora.py` and `tools-py/fireredvad_evaluate_corpora.bat` evaluating AVA-Speech, 15 acoustic noise environments, rock music, singing, and clean speech with automated JSON and Markdown reporting.
- **AVA-Speech Ground Truth Benchmark**: Achieved **80.47% F1-Score** (86.36% Accuracy, 92.74% Specificity) across 420,000 evaluated frames.

### Optimized
- **ESP-DSP Radix-4 Real-Input FFT**: Replaced 1024-float complex FFT (`dsps_fft2r_fc32`) in `esp_firevad_dsp.cpp` with 16-byte aligned 512-float Radix-4 real FFT (`dsps_fft4r_fc32` + `dsps_cplx2real_fc32`), reducing feature buffer RAM by 50% and doubling FFT compute throughput with zero mathematical degradation.
- **Model Directory Alignment**: Standardized model storage under `models/espm` (converted containers) and `models/py` (PyTorch checkpoints).

## 1.0.0 - 2026-11-24

### Added
- **ESPM v3 Container Standard**: Migrated all FireRedVAD models to the standardized ESPM v3 container format (`models/espm/`) matching `esp-idf-model-manager`.
- **64-Byte Tensor Alignment**: Ensured all weight payloads and tensor data are 64-byte aligned for zero-copy memory mapping on ESP32-P4 cache lines.
- **Built-in ROM Model**: Added `firevad_model_data.h` containing an embedded Stream-VAD INT8-CH default model for out-of-the-box operation without requiring an SD card or filesystem.
- **Embedded ESPM Loaders**:
  - `firevad_espm_loader_core.c`: Strict container validation, section CRC verification, typed metadata parsing, and layer binding.
  - `firevad_espm_loader_vfs.c`: Convenient filesystem loader (`esp_firevad_load_espm_file`).
- **Python Exporter & Validator**:
  - `tools-py/fireredvad_models_converter/espm_format.py`: Production-grade ESPM v3 writer with typed `EMET` metadata and `ETIX` v2 tensor indexing.
  - `tools-py/fireredvad_models_converter/fireredvad_verify_espm.py`: Automated CRC32 and alignment validation script.
- **Model Variants**: Pre-converted FP32, INT8 per-tensor, and INT8 per-channel (`INT8-CH`) models for Stream-VAD, Offline VAD, and AED.

### Changed
- **Component Renaming**: Standardized component naming from legacy `esp32-p4-idf-fireredvad` to `esp-idf-fireredvad`.
- **License**: Switched to PolyForm Noncommercial License 1.0.0 with third-party notices for upstream FireRedVAD (Apache 2.0).
- **Target Portability**: Safely guarded ESP32-P4 RISC-V PIE vector assembly with `#if defined(CONFIG_IDF_TARGET_ESP32P4) && defined(__riscv)` and added portable scalar fallbacks.

### Removed
- Removed legacy `.frvd` format and obsolete `examples/.../models_frvd/` folder.
- Cleaned up obsolete AI scratchpad notes, legacy failure logs, and chat dumps from `.docs/`.
