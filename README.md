# FireRedVAD ESP-IDF component

[Changelog](CHANGELOG.md)

This repository contains an ESP-IDF component for running FireRedVAD DFSMN (Deep Feedforward Sequential Memory Network) models on the ESP32-P4. It uses the standardized **ESPM v3** container format and includes a console example with WiFi 6 and audio management for the Waveshare ESP32-P4-WIFI6 development board.

## Component Features

- **DFSMN Neural Architecture**: Native inference engine optimized for causal streaming Voice Activity Detection (Stream-VAD), offline chunked VAD, and Acoustic Event Detection (AED).
- **ESPM v3 Container Standard**: Full integration with `esp-idf-model-manager` and the ESPM v3 specification, featuring typed metadata (`EMET`), tensor index (`ETIX`), CRC32 verification, and 64-byte cacheline-aligned tensor payloads for zero-copy PSRAM execution.
- **Built-in ROM Model**: Includes an embedded default Stream-VAD model (`firevad_model_data.h`) in flash memory, enabling immediate out-of-the-box VAD without requiring an SD card, SPIFFS, or filesystem initialization.
- **Multi-Precision Support**: Pre-converted models in FP32, INT8 per-tensor, and INT8 per-channel (`INT8-CH`) for optimized balance between latency and accuracy.
- **ESP-DSP Radix-4 Real-Input FFT**: Accelerated 80-bin Log-Mel filterbank extraction utilizing 16-byte aligned `dsps_fft4r_fc32` and `dsps_cplx2real_fc32`, cutting FFT memory footprint by 50% and doubling compute throughput over complex Radix-2 FFT with zero mathematical loss.
- **ESP32-P4 Acceleration**: Leverages the ESP32-P4 RISC-V hardware FPU, instruction prefetching (`__builtin_prefetch`), cacheline alignment, and zero-wait-state internal SRAM/TCM for streaming ring-buffer caches.
- **WiFi 6 & Transfer Server**: Example application includes ESP-Hosted SDIO WiFi 6 (ESP32-C6) and embedded HTTP upload server for over-the-air model and audio deployment.

## Hardware and Requirements

- **Target**: ESP32-P4 (e.g. Waveshare ESP32-P4-WIFI6 development board).
- **ESP-IDF**: Version 6.0.2 or later.
- **PSRAM**: Required for FP32/INT8 model weights (P4 boards typically feature 16 MB or 32 MB PSRAM).

## Adding the Component

Add the component to your project's `idf_component.yml`:

```yaml
dependencies:
  esp-idf-fireredvad:
    git: "https://github.com/Strg-Alt-Entf-0x00/esp-idf-fireredvad.git"
    version: "^1.1.0"
```

## Scientific Benchmarks & Validation

FireRedVAD Stream-VAD was benchmarked against the gold-standard AVA-Speech dataset and 15 acoustic noise stress environments using the `autonomous-dev-toolbox` corpus suite:

### 1. AVA-Speech Ground Truth Benchmark (420,000 frames / 70 min audio)

| Metric | Measured Value | Standard Target | Status |
| :--- | :--- | :--- | :--- |
| **F1-Score** | **80.47%** | > 70.0% | **PASS** |
| **Accuracy** | **86.36%** | > 75.0% | **PASS** |
| **Precision** | **86.04%** | > 75.0% | **PASS** |
| **Recall** | **75.58%** | > 65.0% | **PASS** |
| **Specificity** | **92.74%** | > 70.0% | **PASS** |
| **Real-Time Factor (RTF)** | **0.1011** | < 0.20 (5x Realtime) | **PASS** |

### 2. Acoustic Noise & Non-Speech Robustness

- **Rock Music Rejection**: **0.00% False Alarm Rate** (Mean score 0.0690).
- **Severe Acoustic Immunity**: **0.00% False Alarm Rate** under White Noise, Machine Room, Light Fan, Electrical Buzz, Server Fans, Mic Bumps, and Gym Ambience.
- **Overall Noise Immunity**: **78.27% Rejection** across 15 real-world noise environments.

## Quick Start

### 1. Zero-Configuration ROM Default Model

```c
#include "esp_firevad.h"
#include "esp_firevad_dsp.h"

// 1. Initialize DSP frontend (Radix-4 Real-Input FFT)
ESP_ERROR_CHECK(esp_firevad_dsp_init());

// 2. Load the built-in default ROM model (Stream-VAD INT8-CH)
EspFirevadModel model;
ESP_ERROR_CHECK(esp_firevad_load_default_model(&model));

// 3. Process 160-sample (10 ms) audio frames @ 16 kHz
int16_t pcm_160[160]; // capture from I2S
float features_80[80];
float out_prob = 0.0f;

esp_firevad_dsp_extract_features(pcm_160, features_80, NULL);
esp_firevad_infer_frame(&model, features_80, true, &out_prob);

if (out_prob > 0.60f) {
    // Speech detected
}

// 4. Free resources when done
esp_firevad_free(&model);
```

### 2. Dynamic ESPM Model Loading from SD Card / VFS

```c
EspFirevadModel model;
ESP_ERROR_CHECK(esp_firevad_load_espm_file(&model, "/sd/models/firered_stream-vad_int8_ch.espm"));
```

## Available Models (`models/espm/`)

Pre-converted models in the standardized ESPM v3 container format are included in `models/espm/`:

| Model | Variant | Quantization | Size | Description |
|---|---|---|---|---|
| `firered_stream-vad_int8_ch.espm` | Stream-VAD | INT8 per-channel | ~597 KB | **Recommended**: High accuracy, fast execution |
| `firered_stream-vad_int8.espm` | Stream-VAD | INT8 per-tensor | ~585 KB | Compact streaming VAD |
| `firered_stream-vad_fp32.espm` | Stream-VAD | FP32 | ~2.28 MB | Full floating-point reference model |
| `firered_vad_int8_ch.espm` | VAD (Offline) | INT8 per-channel | ~620 KB | Non-causal with lookahead filter |
| `firered_vad_fp32.espm` | VAD (Offline) | FP32 | ~2.35 MB | Offline reference model |
| `firered_aed_int8_ch.espm` | AED | INT8 per-channel | ~620 KB | Audio Event Detection (Speech, Music, Singing) |
| `firered_aed_fp32.espm` | AED | FP32 | ~2.37 MB | Full-precision event detector |

## Tooling & Validation

To download PyTorch checkpoints and re-generate ESPM v3 models:

```cmd
tools-py\fireredvad_download_models_py.bat
tools-py\fireredvad_convert_models.bat
```

To run scientific corpus validation against AVA-Speech and noise corpora:

```cmd
tools-py\fireredvad_evaluate_corpora.bat
```

To verify the integrity and CRC compliance of all exported ESPM v3 models:

```cmd
python tools-py\fireredvad_models_converter\fireredvad_verify_espm.py
```

## Example Application

A full-featured console application is available under `examples/ws-esp32-p4-wifi6-console-vad-dev`:
- Real-time microphone capture via ES8311 I2S codec
- Background noise auto-calibration
- WiFi 6 transfer server for uploading audio and `.espm` models to `/sd/`
- Interactive console with commands: `threshold`, `test_wav`, `load_model`, `record`, `benchmark`, `wifi`

Build the example with:
```cmd
cd examples\ws-esp32-p4-wifi6-console-vad-dev
esp-build.bat build
```

## License and Attribution

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

The FireRedVAD neural network architecture, model weights, and CMVN statistics originate from [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) by Xiaohongshu (FireRedTeam), licensed under Apache License 2.0:
Copyright (c) 2024 FireRedTeam.
