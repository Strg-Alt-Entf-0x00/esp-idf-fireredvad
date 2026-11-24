#include "esp_firevad_dsp.h"
#include "mel_constants.h"
#include "dsps_fft4r.h"
#include "dsps_math.h"
#include <string.h>
#include <math.h>

#ifdef ESP_PLATFORM
#include "esp_log.h"
#include "esp_cpu.h"
#include "esp_attr.h"
static const char* TAG = "ESP_FIREVAD_DSP";
#endif

esp_err_t esp_firevad_dsp_init(void) {
    // 512-point real FFT packed into 256 complex values via Radix-4
    esp_err_t ret = dsps_fft4r_init_fc32(NULL, 256);
#ifdef ESP_PLATFORM
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize Radix-4 Real FFT: %d", ret);
    }
#endif
    return ret;
}

#ifdef ESP_PLATFORM
#define DSP_IRAM_ATTR IRAM_ATTR
#else
#define DSP_IRAM_ATTR
#endif

static inline float fast_log_mel_energy(float energy) {
    if (energy < 1e-6f) return -13.815510557964274f;

    uint32_t ux;
    memcpy(&ux, &energy, sizeof(ux));
    int32_t iexp = (int32_t)(ux >> 23) - 127;
    ux = (ux & 0x007FFFFFu) | 0x3F800000u;
    float mf;
    memcpy(&mf, &ux, sizeof(mf));

    float tk = (mf - 1.0f) / (mf + 1.0f);
    float tk2 = tk * tk;
    float ln_m = tk * (2.0f + tk2 * (0.666667f + tk2 * (0.4f + tk2 * 0.285714f)));
    return ln_m + (float)iexp * 0.693147180f;
}

static DRAM_ATTR float window_buffer[400] = {0};
static DRAM_ATTR float fft_buf[512] __attribute__((aligned(16))) = {0};
static DRAM_ATTR float power_spectrum[257] __attribute__((aligned(16))) = {0};
static float prev_samp = 0.0f;

void esp_firevad_dsp_reset(void) {
    memset(window_buffer, 0, sizeof(window_buffer));
    memset(fft_buf, 0, sizeof(fft_buf));
    memset(power_spectrum, 0, sizeof(power_spectrum));
    prev_samp = 0.0f;
}

void DSP_IRAM_ATTR esp_firevad_dsp_extract_features(const int16_t* pcm_160, float* features_80, float* out_energy) {
    memmove(window_buffer, window_buffer + 160, (400 - 160) * sizeof(float));
    float total_sq = 0.0f;

    // 1. Append raw PCM samples to window buffer
    for (int i = 0; i < 160; i++) {
        float raw = (float)pcm_160[i];
        if (raw > 32767.0f) raw = 32767.0f;
        if (raw < -32768.0f) raw = -32768.0f;
        window_buffer[240 + i] = raw;
        total_sq += raw * raw;
    }

    if (out_energy) {
        *out_energy = sqrtf(total_sq / 160.0f);
    }

    // 2. Compute frame mean (Kaldi remove_dc_offset)
    float frame_mean = 0.0f;
    for (int i = 0; i < 400; i++) {
        frame_mean += window_buffer[i];
    }
    frame_mean /= 400.0f;

    // 3. Pre-emphasis and windowing directly into packed real FFT buffer
    if (prev_samp == 0.0f) {
        prev_samp = window_buffer[0] - frame_mean;
    }
    for (int i = 0; i < 400; i++) {
        float curr_samp = window_buffer[i] - frame_mean;
        float preemph_samp = curr_samp - 0.97f * prev_samp;
        prev_samp = curr_samp;
        fft_buf[i] = preemph_samp * KALDI_WINDOW[i];
    }
    memset(&fft_buf[400], 0, (512 - 400) * sizeof(float));

    // 4. Radix-4 Real-Input FFT (512-point real input packed into 256 complex bins)
    dsps_fft4r_fc32(fft_buf, 256);
    dsps_bit_rev4r_fc32(fft_buf, 256);
    dsps_cplx2real_fc32(fft_buf, 256);

    // DC and Nyquist components
    power_spectrum[0] = fft_buf[0] * fft_buf[0];
    power_spectrum[256] = fft_buf[1] * fft_buf[1];

    // Intermediate frequency bins
    for (int i = 1; i < 256; i++) {
        float re = fft_buf[i * 2];
        float im = fft_buf[i * 2 + 1];
        power_spectrum[i] = re * re + im * im;
    }

    // 5. Apply Kaldi Sparse Mel Filterbank (80 bins)
    for (int m = 0; m < 80; m++) {
        const uint8_t cnt = KALDI_MEL_COUNTS[m];
        const uint16_t off = KALDI_MEL_OFFSETS[m];
        const uint16_t* idx_ptr = KALDI_MEL_INDICES + off;
        const float* weight_ptr = KALDI_MEL_WEIGHTS + off;

        float mel_energy = 0.0f;
        for (int i = 0; i < cnt; i++) {
            mel_energy += power_spectrum[idx_ptr[i]] * weight_ptr[i];
        }

        features_80[m] = fast_log_mel_energy(mel_energy);
    }
}
