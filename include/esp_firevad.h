#pragma once
/**
 * FireRedVAD Native Inference Engine for ESP32-P4
 *
 * Runs DFSMN (Deep Feedforward Sequential Memory Network) inference
 * on 80-bin log-mel features with streaming ring-buffer caches.
 * Supports ESPM v3 container models (FP32, INT8, INT8-CH).
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Architecture parameters defining the exact network topology.
 */
typedef struct {
    uint32_t D;      // Input dimension (80 = mel bins)
    uint32_t H;      // Hidden size (256)
    uint32_t P;      // Projection size (128)
    uint32_t odim;   // Output dimension (1 for VAD, 2 for AED)
    uint32_t R;      // Number of DFSMN blocks (8)
    uint32_t M;      // Number of DNN layers (1)
    uint32_t N1;     // Lookback order (20)
    uint32_t S1;     // Lookback stride (1)
    uint32_t N2;     // Lookahead order (0 for streaming)
    uint32_t S2;     // Lookahead stride
} EspFirevadArchParams;

/**
 * Pointers into the weight buffer for a single Dense layer (weight + bias).
 */
typedef struct {
    const void* weight;          // [out_dim, in_dim] row-major (float* or int8_t*)
    const void* bias;            // [out_dim] float32 (or NULL if no bias)
    float weight_scale;          // For global scale
    float bias_scale;            // For global scale
    const float* channel_scales; // For per-channel scale (array of size out_dim)
    uint32_t in_dim;
    uint32_t out_dim;
} DenseLayer;

/**
 * Pointers into the weight buffer for a single FSMN memory filter.
 */
typedef struct {
    const void* lookback_weight;   // [N1, P] depthwise conv kernel
    float lookback_scale;
    const void* lookahead_weight;  // [N2, P] or NULL
    float lookahead_scale;
} FsmnFilter;

/**
 * Complete model state: architecture + weight pointers + streaming cache.
 */
typedef struct {
    uint32_t model_type; // 0=VAD, 1=Stream-VAD, 2=AED
    bool is_int8;
    bool is_int16;
    bool is_int8_per_ch;

    // Architecture config
    EspFirevadArchParams arch;
    int version;

    // CMVN normalization
    uint32_t cmvn_dim;
    const float* cmvn_means;
    const float* cmvn_istd;

    // Layer pointers (into weight_buffer)
    DenseLayer fc1;           // Input projection: D -> H
    DenseLayer fc2;           // Second projection: H -> P
    FsmnFilter fsmn1;         // First FSMN block filter

    // R-1 DFSMN blocks
    DenseLayer* block_fc1;    // Array[R-1]: P -> H
    DenseLayer* block_fc2;    // Array[R-1]: H -> P (no bias)
    FsmnFilter* block_fsmn;   // Array[R-1]: memory filters

    // DNN layers (after DFSMN blocks)
    DenseLayer* dnn_layers;   // Array[M]
    uint32_t num_dnn_layers;

    // Output layer
    DenseLayer out;           // H -> odim

    // Streaming cache: one ring buffer per FSMN block
    float** fsmn_caches;        // Array[R] of cache buffers
    uint32_t* fsmn_cache_heads; // Array[R] of write indices
    uint32_t cache_len;         // (N1-1)*S1

    // Raw weight buffer (allocated in PSRAM)
    uint8_t* weight_buffer;
    size_t weight_buffer_size;
    bool owns_weight_buffer;

    void** tensor_ptrs;
    uint32_t num_tensors;
    uint32_t max_tensors;

    // Scratch buffers for inference
    float* scratch_h;            // [H]
    float* scratch_p;            // [P]
    float* scratch_p2;           // [P]
    float* scratch_conv;         // [P] for conv output
    int8_t* scratch_in_q;        // [max(H, P)] for Int8 dynamic quant
    int16_t* scratch_in_q16;     // [max(H, P)] for Int16 dynamic quant
    float* chunk_scratch;        // reusable scratch area for infer_chunk
    size_t chunk_scratch_bytes;
} EspFirevadModel;

// ---- Public Model Loading API ----

/**
 * @brief Load an ESPM v3 model from an in-memory buffer.
 *        Takes ownership of raw_data on success, or frees it on failure.
 *
 * @param model Output: populated model struct.
 * @param raw_data Pointer to complete .espm binary in memory.
 * @param file_size Exact file size in bytes.
 * @return esp_err_t ESP_OK on success, error code otherwise.
 */
esp_err_t esp_firevad_load_espm(EspFirevadModel* model, uint8_t* raw_data, size_t file_size);

/**
 * @brief Load an ESPM v3 model from a mounted VFS filesystem path (SD card, SPIFFS).
 *
 * @param model Output: populated model struct.
 * @param espm_path Absolute path to the .espm file.
 * @return esp_err_t ESP_OK on success, error code otherwise.
 */
esp_err_t esp_firevad_load_espm_file(EspFirevadModel* model, const char* espm_path);

/**
 * @brief Load the built-in default ROM model (Stream-VAD INT8-CH).
 *        No external file or SD card required.
 *
 * @param model Output: populated model struct.
 * @return esp_err_t ESP_OK on success, error code otherwise.
 */
esp_err_t esp_firevad_load_default_model(EspFirevadModel* model);

/**
 * @brief Legacy loader for .frvd binary format.
 */
int esp_firevad_load(const uint8_t* data, size_t data_len, EspFirevadModel* model);

// ---- Inference API ----

/**
 * @brief Run streaming inference on a single frame of 80-dim log-mel features.
 *
 * @param model Loaded model.
 * @param features Input features [D] (80 floats, fbank-extracted).
 * @param apply_cmvn If true, apply CMVN normalization.
 * @param out_probs Output array of size `odim`.
 */
void esp_firevad_infer_frame(EspFirevadModel* model, const float* features, bool apply_cmvn, float* out_probs);

/**
 * @brief DUAL-CORE API: Extract features only (for Core 0).
 */
void esp_firevad_prepare_features(EspFirevadModel* model, const float* features, bool apply_cmvn, float* out_features);

/**
 * @brief DUAL-CORE API: Run inference on pre-extracted features (for Core 1).
 */
void esp_firevad_infer_prepared(EspFirevadModel* model, const float* features, float* out_probs);

/**
 * @brief Run inference on a full chunk of 80-dim log-mel features.
 */
void esp_firevad_infer_chunk(EspFirevadModel* model, const float* features, uint32_t num_frames, bool apply_cmvn, float* out_probs);

/**
 * @brief Reset all streaming ring-buffer caches to zero.
 */
void esp_firevad_reset(EspFirevadModel* model);

/**
 * @brief Free all allocated memory (weights, caches, scratch buffers).
 */
void esp_firevad_free(EspFirevadModel* model);

/**
 * @brief Get total memory usage in bytes (weights + caches + scratch).
 */
size_t esp_firevad_memory_usage(const EspFirevadModel* model);

#ifdef __cplusplus
}
#endif
