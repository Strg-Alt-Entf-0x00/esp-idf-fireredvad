# FireVAD Console Example (ESP32-P4)

This is the primary interactive evaluation and test environment for the FireRedVAD pipeline on the **Waveshare ESP32-P4-WIFI6** (Chip Rev v1.3).

The interactive console (REPL) allows you to dynamically load ESPM v3 neural network models, stream audio from digital (`INMP441`) or analog (`ES8311`) microphones, record test WAV files to an SD card, configure WiFi 6 for wireless model deployment, and tune Automatic Gain Control (AGC) and Pre-VAD thresholds in real-time.

## Prerequisites

- **ESP-IDF**: Version 6.0.2 or later (required for ESP32-P4 multi-core routing and RISC-V PIE support)
- **Target Hardware**: Waveshare ESP32-P4-WIFI6 development board
- **Storage**: MicroSD card formatted as FAT32 (optional when using the built-in ROM model)

### Microphone Options

#### INMP441 Digital I2S Microphone (Recommended)
- **Pins**: BCLK=GPIO 20, WS=GPIO 21, DIN=GPIO 22
- **Signal-to-Noise Ratio (SNR)**: 23.2 dB
- **Noise Floor**: -67.99 dBFS
- **Sensitivity at 1m**: -44.73 dBFS
- **Key Characteristics**:
  - Direct digital I2S capture with high immunity to PCB EMI and WiFi RF interference
  - Low noise floor without analog DC-offset or pop artifacts
  - Expandable to stereo capture and beamforming by sharing the I2S bus (toggling L/R pin)

#### ES8311 Onboard Analog Microphone (Fallback)
- **Interface**: I2C control (SDA=GPIO 7, SCL=GPIO 8) + I2S0 audio (MCLK=GPIO 13, BCLK=GPIO 12, WS=GPIO 10, DIN=GPIO 11, DOUT=GPIO 9)
- **Signal-to-Noise Ratio (SNR)**: 20.7 dB (at maximum 42 dB PGA gain)
- **Noise Floor**: -52.67 dBFS (requires gain boost)
- **Sensitivity at 1m**: -31.91 dBFS (with 42 dB gain)
- **Key Characteristics**:
  - Analog circuit routed across PCB layers; more susceptible to display and RF noise
  - Requires hardware PGA gain boost (36 to 42 dB) for distant speech detection
  - Integrated 80 Hz high-pass filter mitigates power-on DC offset

## 1. Prepare AI Models

FireRedVAD supports models in the standardized **ESPM v3** container format. You have three ways to run models:

### Option A: Built-in ROM Default Model (Zero Configuration)
No SD card or model download is required. The firmware embeds a production-ready Stream-VAD INT8-CH model in flash ROM. Simply boot and enter:
```bash
vad_model_load default
```

### Option B: Copy Pre-Converted Models to MicroSD Card
Pre-converted ESPM v3 models are included in the repository under `models/espm/`:
- `firered_stream-vad_int8_ch.espm` (Recommended: ~597 KB, per-channel quantization)
- `firered_stream-vad_int8.espm` (~585 KB, per-tensor quantization)
- `firered_stream-vad_fp32.espm` (~2.28 MB, floating-point reference)
- `firered_vad_int8_ch.espm` (~620 KB, offline non-causal VAD)
- `firered_aed_int8_ch.espm` (~620 KB, acoustic event detector)

Format a microSD card as FAT32, create a `models` folder on the card, and copy the desired `.espm` files into `/sd/models/`.

### Option C: Wireless Deployment over WiFi 6
Upload models directly to the board over WiFi without removing the SD card (see [WiFi 6 File Transfer Server](#wifi-6-file-transfer-server)).

## 2. Build and Flash

From this directory, build and flash using the automated Windows build script:

```cmd
esp-build.bat build
esp-build.bat flash
esp-build.bat monitor
```

Or using native ESP-IDF tools:

```bash
idf.py set-target esp32p4
idf.py build flash monitor
```

> [!IMPORTANT]
> **Boot Calibration:** Upon boot, the firmware performs a 1-second background acoustic noise calibration. **Remain silent when `PLEASE BE SILENT - CALIBRATING MICROPHONE...` is displayed.** The system will play a chime sequence when calibration is complete.

## 3. Interactive Console Reference

Once booted, the interactive command prompt (`firevad>`) is ready. Type `help` to list all available commands.

### Quick Start Workflow

1. Select the digital microphone:
   ```bash
   mic_select inmp441
   ```
2. Enable Automatic Gain Control:
   ```bash
   agc_enable 1
   ```
3. Load the streaming model (ROM default or from SD card):
   ```bash
   vad_model_load default
   # or: vad_model_load models/firered_stream-vad_int8_ch.espm
   ```
4. Start real-time microphone VAD for 30 seconds:
   ```bash
   vad_infer_mic 30
   ```
5. View inference latency metrics:
   ```bash
   vad_metrics
   ```

### Full Command Reference

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `vad_model_load` | `<file.espm \| default>` | Loads an ESPM v3 model from `/sd/<file.espm>` or the embedded ROM default model into PSRAM. |
| `vad_model_info` | None | Displays architecture topology, quantization mode, frame cache length, and memory consumption. |
| `vad_cascade_load` | `<stream.espm> <offline.espm>` | Loads dual-core cascade pipeline: fast Stream-VAD gatekeeper on Core 0 and offline classifier on Core 1. |
| `vad_threshold` | `<0.0 - 1.0>` | Sets or inspects the speech detection probability threshold (default: `0.60`). |
| `vad_infer_mic` | `[seconds]` | Runs real-time microphone VAD inference with live ASCII energy and detection visualization. |
| `vad_infer_wav` | `<filename.wav>` | Runs offline VAD inference over an audio file stored on `/sd/`. |
| `vad_metrics` | None | Displays execution latency breakdown, average inference duration, and CPU cycles. |
| `vad_calibrate` | `[seconds]` | Measures ambient noise to calibrate pre-VAD energy gate thresholds. |
| `vad_pre_vad` | `<multiplier>` | Sets pre-VAD energy gate multiplier (`0.0` to disable and rely solely on neural network). |
| `vad_sleep_mode` | `[<interval_ms>]` | Tests software duty-cycling VAD mode with automated light sleep. |
| `vad_dump_golden` | `<filename.wav>` | Dumps intermediate tensor buffers (PCM, log-mel filterbanks, CMVN, probabilities) for golden tests. |
| `vad_test_pie` | `[length]` | Validates ESP32-P4 RISC-V PIE vector SIMD dot-product operations against scalar reference math. |
| `benchmark` | `[num_frames]` | Benchmarks inference performance, Real-Time Factor (RTF), and PSRAM/SRAM footprint. |
| `mic_select` | `<inmp441 \| es8311>` | Selects the active microphone hardware source. |
| `mic_info` | None | Prints microphone hardware configuration, active pins, sample rate, and PGA gain settings. |
| `mic_gain` | `[-s <mult>] [-h <0-11>]` | Adjusts software digital gain multiplier and/or ES8311 hardware PGA gain stage. |
| `mic_level` | None | Samples ambient room acoustic energy for 1 second, printing peak and RMS dBFS levels. |
| `record_mic` | `<file.wav> <seconds>` | Records raw audio from the active microphone into a standard 16 kHz 16-bit WAV file on `/sd/`. |
| `speaker_vol` | `<0 - 100>` | Adjusts ES8311 onboard speaker amplifier output volume. |
| `agc_enable` | `<0 \| 1>` | Enables (`1`) or disables (`0`) the dynamic software Automatic Gain Control. |
| `agc_info` | None | Displays AGC runtime status, current gain multiplier, and target RMS window. |
| `agc_config` | `<tgt> <max> <min> <gate>` | Configures AGC parameters (target RMS, maximum gain, minimum gain, noise gate threshold). |
| `ls` / `fs_ls` | `[path]` | Lists directory contents and file sizes on the mounted FAT32 SD filesystem (`/sd/`). |
| `wifi` | `<subcommand>` | Manages WiFi 6 station connection, NVS credentials, and HTTP file transfer server. |
| `guide` | None | Displays the recommended quick-start test sequence. |
| `help` | None | Displays full list of available ESP-IDF system and component console commands. |

## 4. WiFi 6 File Transfer Server

The console integrates ESP-Hosted SDIO communication with the onboard ESP32-C6 companion chip to provide WiFi 6 connectivity and a background HTTP file server on port `8080`.

### Wi-Fi Configuration

1. **Configure and connect to an Access Point:**
   ```bash
   wifi setup "MySSID" "MyPassword"
   ```
   Credentials are saved in non-volatile storage (NVS) for automatic reconnection on subsequent boots.

2. **Check connection status:**
   ```bash
   wifi status
   ```
   When connected, the output shows the allocated IP address and API endpoint.

3. **Retrieve the transfer authorization token:**
   ```bash
   wifi token
   ```
   Each boot generates a secure session bearer token required for HTTP requests.

4. **Disconnect or clear saved credentials:**
   ```bash
   wifi disconnect
   wifi forget
   ```

### REST API Endpoints

The HTTP server provides file management for three directories on the SD card:
- `/api/models/<filename>` -> mapped to `/sd/models/<filename>`
- `/api/audio/<filename>` -> mapped to `/sd/audio/<filename>`
- `/api/results/<filename>` -> mapped to `/sd/results/<filename>`

All requests require the `Authorization: Bearer <TOKEN>` header.

#### Upload an ESPM Model from your PC
```bash
curl -X PUT -H "Authorization: Bearer <TOKEN>" \
  --data-binary @models/espm/firered_stream-vad_int8_ch.espm \
  http://<ESP32_IP>:8080/api/models/firered_stream-vad_int8_ch.espm
```

#### Upload a Test WAV Audio File
```bash
curl -X PUT -H "Authorization: Bearer <TOKEN>" \
  --data-binary @test_speech.wav \
  http://<ESP32_IP>:8080/api/audio/test_speech.wav
```

#### Download a Recorded Audio File
```bash
curl -H "Authorization: Bearer <TOKEN>" \
  -o downloaded_recording.wav \
  http://<ESP32_IP>:8080/api/audio/recording.wav
```

#### List Files in a Directory
```bash
curl -H "Authorization: Bearer <TOKEN>" \
  http://<ESP32_IP>:8080/api/models
```

## 5. MicroSD Card Organization

The firmware expects the following directory structure on the FAT32 microSD card:

```text
/sd/
|-- models/
|   |-- firered_stream-vad_int8_ch.espm
|   |-- firered_stream-vad_int8.espm
|   |-- firered_stream-vad_fp32.espm
|   |-- firered_vad_int8_ch.espm
|   `-- firered_aed_int8_ch.espm
|-- audio/
|   |-- speech_sample.wav
|   `-- noise_sample.wav
`-- results/
    `-- eval_output.csv
```

Directories are automatically created upon initial boot if they do not already exist.
