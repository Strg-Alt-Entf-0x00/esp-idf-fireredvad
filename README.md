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

**Evaluation Date**: 2026-11-24 11:50:17  
**Evaluated Model**: FireRedVAD Stream-VAD (DFSMN, R=8, M=1, H=256, P=128, N1=20, S1=1)  
**Container Format**: ESPM v3 (64-byte cacheline aligned, zero-copy model manager)  
**Operating Threshold**: 0.60  

FireRedVAD was benchmarked against the gold-standard AVA-Speech corpus (420,000 frames / 70 minutes of real-world audio) and 15 acoustic noise stress environments:

### 1. Executive Benchmark Summary

| Metric | Measured Value | Standard Target | Status |
| :--- | :--- | :--- | :--- |
| **AVA Speech F1-Score** | **80.47%** | > 70.0% | **PASS** |
| **AVA Speech Accuracy** | **86.36%** | > 75.0% | **PASS** |
| **AVA Speech Precision** | **86.04%** | > 75.0% | **PASS** |
| **AVA Speech Recall** | **75.58%** | > 65.0% | **PASS** |
| **AVA Speech Specificity** | **92.74%** | > 70.0% | **PASS** |
| **Noise Rejection Rate** | **78.27%** | > 70.0% | **PASS** |
| **Mean Inference Latency** | **1011.0 µs** (1.01 ms) | < 10,000 µs (10 ms) | **PASS** |
| **Median (P50) Latency** | **938.1 µs** (0.94 ms) | < 5,000 µs | **PASS** |
| **95th Percentile (P95)** | **1495.2 µs** (1.50 ms) | < 10,000 µs | **PASS** |
| **Throughput** | **996.0 frames/sec** | > 100 fps | **PASS** |
| **Real-Time Factor (RTF)** | **0.1011** (~10x Realtime) | < 0.20 (5x Realtime) | **PASS** |
| **ESPM v3 Container Parity** | **9 / 9 Models Valid** | 100% | **PASS** |

### 2. AVA-Speech Ground Truth Breakdown (420,000 Frames / 70 Min Audio)

- **Total Frames Evaluated**: 420,000 (4,200.0 seconds)
- **True Positives (TP)**: 118,014
- **False Positives (FP)**: 19,145
- **True Negatives (TN)**: 244,710
- **False Negatives (FN)**: 38,131

| Clip File | Duration | F1-Score | Accuracy | Precision | Recall | Latency | RTF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `human_xO4ABy2iOQA_clip.wav` | 300.0s | 84.02% | 85.98% | 87.33% | 80.95% | 973.4 µs | 0.0973 |
| `human_y7ncweROe9U_clip.wav` | 300.0s | 84.06% | 82.55% | 96.71% | 74.33% | 1087.1 µs | 0.1087 |
| `human_-5KQ66BBWC4_clip.wav` | 900.0s | 76.58% | 90.47% | 76.01% | 77.16% | 939.7 µs | 0.0940 |
| `human_-FaXLcSFjUI_clip.wav` | 900.0s | 63.91% | 88.19% | 61.53% | 66.49% | 953.6 µs | 0.0954 |
| `human_-IELREHX_js_clip.wav` | 900.0s | 81.58% | 88.40% | 92.08% | 73.24% | 1170.2 µs | 0.1170 |
| `human_053oq2xB3oU_clip.wav` | 900.0s | 83.67% | 79.79% | 90.71% | 77.65% | 942.2 µs | 0.0942 |

### 3. Acoustic Noise Robustness (15 Environments)

| Acoustic Environment | Duration | False Alarms | False Alarm Rate | Max Prob | Mean Prob |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `noise-constant-white.wav` | 5.0s | 0 | **0.00%** | 0.2754 | 0.0319 |
| `noise-electrical-buzz.wav` | 5.0s | 0 | **0.00%** | 0.1503 | 0.0174 |
| `noise-gym-ambience.wav` | 5.0s | 0 | **0.00%** | 0.3020 | 0.0706 |
| `noise-light-fan.wav` | 5.0s | 0 | **0.00%** | 0.4288 | 0.0241 |
| `noise-machine-room.wav` | 5.0s | 0 | **0.00%** | 0.3157 | 0.0375 |
| `noise-mic-bumps.wav` | 5.0s | 0 | **0.00%** | 0.4640 | 0.0372 |
| `noise-server-fans.wav` | 5.0s | 0 | **0.00%** | 0.4937 | 0.0582 |
| `noise-mic-static.wav` | 5.0s | 3 | **0.60%** | 0.7954 | 0.0445 |
| `noise-clock-tick.wav` | 5.0s | 5 | **1.00%** | 0.8219 | 0.0230 |
| `noise-laundry-machine.wav` | 5.0s | 8 | **1.60%** | 0.8612 | 0.0725 |
| `noise-cafeteria.wav` | 5.0s | 137 | **27.40%** | 0.9839 | 0.4215 |
| `noise-office-ambience.wav` | 5.0s | 190 | **38.00%** | 0.9549 | 0.4334 |
| `noise-pizzeria.wav` | 5.0s | 302 | **60.40%** | 0.9929 | 0.6282 |
| `noise-babble-small-room.wav` | 5.0s | 489 | **97.80%** | 0.9971 | 0.9374 |
| `noise-babble-english.wav` | 5.0s | 496 | **99.20%** | 0.9975 | 0.9666 |

### 4. Non-Speech & Acoustic Rejection

| Category | Audio Clip | Duration | False Alarm Rate | Mean Probability |
| :--- | :--- | :--- | :--- | :--- |
| **Rock Music** | `music-rock.wav` | 8.8s | **0.00%** | 0.0690 |
| **Nature / Birds** | `negative-birds.wav` | 15.0s | **3.53%** | 0.1280 |
| **Singing / Vocal** | `singing-vocal.wav` | 14.1s | **59.07%** | 0.5930 |

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

To verify the integrity and CRC compliance of all exported ESPM v3 models:

```cmd
python tools-py\fireredvad_models_converter\fireredvad_verify_espm.py
```

## Example Application

A full-featured console application is available under `examples/ws-esp32-p4-wifi6-console-vad-dev`:
- Real-time microphone capture via INMP441 (digital I2S, 23.2 dB SNR) or onboard ES8311 analog codec
- Background noise auto-calibration and pre-VAD energy gating
- Embedded default ROM model support (`vad_model_load default`) requiring no SD card or network setup
- ESP-Hosted SDIO WiFi 6 transfer server for uploading audio and `.espm` models directly to `/sd/`
- Dual-core cascade architecture (`vad_cascade_load`) running fast gatekeeper on Core 0 and offline classifier on Core 1
- Interactive console with commands: `vad_model_load`, `vad_infer_mic`, `vad_infer_wav`, `vad_cascade_load`, `vad_metrics`, `benchmark`, `record_mic`, `agc_enable`, `mic_select`, `wifi`

Build the example with:
```cmd
cd examples\ws-esp32-p4-wifi6-console-vad-dev
esp-build.bat build
```

## License and Attribution

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

The FireRedVAD neural network architecture, model weights, and CMVN statistics originate from [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) by Xiaohongshu (FireRedTeam), licensed under Apache License 2.0:
Copyright (c) 2024 FireRedTeam.
