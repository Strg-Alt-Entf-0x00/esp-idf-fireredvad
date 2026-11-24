/**
 * @file firevad_espm_loader_core.c
 * @brief Core ESPM v3 container parser and tensor binder for FireRedVAD.
 */

#include "firevad_espm_loader.h"
#include "esp_model_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

#include "esp_log.h"
#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#endif

static const char *TAG = "firevad_loader";

typedef struct {
    const uint8_t *data;
    size_t size;
} buffer_context_t;

static esp_err_t buffer_read_at(void *context, uint64_t offset, void *buffer, size_t size)
{
    const buffer_context_t *ctx = (const buffer_context_t *)context;
    if (!ctx || offset > ctx->size || size > ctx->size - offset) {
        return ESP_ERR_INVALID_SIZE;
    }
    memcpy(buffer, ctx->data + (size_t)offset, size);
    return ESP_OK;
}

static const esp_model_tensor_record_v2_t *find_tensor(
    const esp_model_tensor_record_v2_t *records,
    uint32_t count,
    const char *name)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (strcmp(records[i].name, name) == 0) {
            return &records[i];
        }
    }
    return NULL;
}

static void* firevad_alloc_aligned(size_t size, uint32_t caps)
{
#ifdef ESP_PLATFORM
    void *ptr = heap_caps_aligned_alloc(16, size, caps);
    if (!ptr && (caps & MALLOC_CAP_INTERNAL)) {
        ptr = heap_caps_aligned_alloc(16, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }
    return ptr;
#else
    return malloc(size);
#endif
}

static void firevad_free_aligned(void *ptr)
{
#ifdef ESP_PLATFORM
    heap_caps_free(ptr);
#else
    free(ptr);
#endif
}

static void bind_dense_layer(
    DenseLayer *layer,
    const esp_model_tensor_record_v2_t *w_rec,
    const esp_model_tensor_record_v2_t *b_rec,
    const uint8_t *tensor_data,
    size_t tensor_data_size,
    uint32_t in_dim,
    uint32_t out_dim)
{
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->weight = NULL;
    layer->bias = NULL;
    layer->weight_scale = 1.0f;
    layer->bias_scale = 1.0f;
    layer->channel_scales = NULL;

    if (w_rec) {
        if (w_rec->data_offset + w_rec->data_size <= tensor_data_size) {
            layer->weight = tensor_data + w_rec->data_offset;
        }
        if (w_rec->quant_type == ESPM_TENSOR_QUANT_SYMMETRIC_AXIS && w_rec->scales_size > 0) {
            if (w_rec->scales_offset + w_rec->scales_size <= tensor_data_size) {
                layer->channel_scales = (const float *)(tensor_data + w_rec->scales_offset);
            }
        } else if (w_rec->quant_type == ESPM_TENSOR_QUANT_SYMMETRIC && w_rec->scales_size >= sizeof(float)) {
            if (w_rec->scales_offset + sizeof(float) <= tensor_data_size) {
                memcpy(&layer->weight_scale, tensor_data + w_rec->scales_offset, sizeof(float));
            }
        }
    }

    if (b_rec) {
        if (b_rec->data_offset + b_rec->data_size <= tensor_data_size) {
            layer->bias = tensor_data + b_rec->data_offset;
        }
    }
}

static void bind_fsmn_filter(
    FsmnFilter *filter,
    const esp_model_tensor_record_v2_t *lb_rec,
    const esp_model_tensor_record_v2_t *la_rec,
    const uint8_t *tensor_data,
    size_t tensor_data_size)
{
    filter->lookback_weight = NULL;
    filter->lookback_scale = 1.0f;
    filter->lookahead_weight = NULL;
    filter->lookahead_scale = 1.0f;

    if (lb_rec) {
        if (lb_rec->data_offset + lb_rec->data_size <= tensor_data_size) {
            filter->lookback_weight = tensor_data + lb_rec->data_offset;
        }
        if (lb_rec->scales_size >= sizeof(float) &&
            lb_rec->scales_offset + sizeof(float) <= tensor_data_size) {
            memcpy(&filter->lookback_scale, tensor_data + lb_rec->scales_offset, sizeof(float));
        }
    }

    if (la_rec) {
        if (la_rec->data_offset + la_rec->data_size <= tensor_data_size) {
            filter->lookahead_weight = tensor_data + la_rec->data_offset;
        }
        if (la_rec->scales_size >= sizeof(float) &&
            la_rec->scales_offset + sizeof(float) <= tensor_data_size) {
            memcpy(&filter->lookahead_scale, tensor_data + la_rec->scales_offset, sizeof(float));
        }
    }
}

esp_err_t firevad_load_model_espm_buffer(EspFirevadModel *model, uint8_t *raw_data, size_t file_size)
{
    if (!model || !raw_data || file_size < ESPM_HEADER_SIZE) {
        if (raw_data) firevad_free_aligned(raw_data);
        return ESP_ERR_INVALID_ARG;
    }

    memset(model, 0, sizeof(EspFirevadModel));

    buffer_context_t buf_ctx = { .data = raw_data, .size = file_size };
    esp_model_io_t io = {
        .context = &buf_ctx,
        .read_at = buffer_read_at,
        .file_size = file_size,
    };

    esp_model_container_t container;
    esp_err_t err = esp_model_container_open(&io, &container);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open ESPM container (error %d)", err);
        firevad_free_aligned(raw_data);
        return err;
    }

    if (container.header.model_type != MODEL_TYPE_VAD &&
        container.header.model_type != MODEL_TYPE_CUSTOM) {
        ESP_LOGE(TAG, "Unexpected model type: %u", container.header.model_type);
        firevad_free_aligned(raw_data);
        return ESP_ERR_NOT_SUPPORTED;
    }

    // Locate required sections
    const esp_model_section_t *sec_idx = esp_model_container_find_section(&container, SECTION_TYPE_TENSOR_INDEX);
    const esp_model_section_t *sec_data = esp_model_container_find_section(&container, SECTION_TYPE_TENSOR_DATA);
    const esp_model_section_t *sec_meta = esp_model_container_find_section(&container, SECTION_TYPE_METADATA);

    if (!sec_idx || !sec_data || !sec_meta) {
        ESP_LOGE(TAG, "Missing required ESPM sections (idx=%p, data=%p, meta=%p)",
                 sec_idx, sec_data, sec_meta);
        firevad_free_aligned(raw_data);
        return ESP_ERR_NOT_FOUND;
    }

    // Verify section payloads
    if (esp_model_container_verify_section_buffer(sec_idx, raw_data + sec_idx->offset, sec_idx->size) != ESP_OK ||
        esp_model_container_verify_section_buffer(sec_data, raw_data + sec_data->offset, sec_data->size) != ESP_OK ||
        esp_model_container_verify_section_buffer(sec_meta, raw_data + sec_meta->offset, sec_meta->size) != ESP_OK) {
        ESP_LOGE(TAG, "ESPM section CRC check failed");
        firevad_free_aligned(raw_data);
        return ESP_ERR_INVALID_CRC;
    }

    // Parse metadata
    esp_model_metadata_view_t meta_view;
    err = esp_model_metadata_open(raw_data + sec_meta->offset, sec_meta->size, &meta_view);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open ESPM metadata view (error %d)", err);
        firevad_free_aligned(raw_data);
        return err;
    }

    const char *arch_str = NULL;
    size_t arch_str_len = 0;
    if (esp_model_metadata_get_utf8(&meta_view, "general.architecture", &arch_str, &arch_str_len) != ESP_OK ||
        arch_str_len != strlen("fireredvad.dfsmn") ||
        memcmp(arch_str, "fireredvad.dfsmn", arch_str_len) != 0) {
        ESP_LOGE(TAG, "Unsupported model architecture (expected fireredvad.dfsmn)");
        firevad_free_aligned(raw_data);
        return ESP_ERR_NOT_SUPPORTED;
    }

    esp_model_metadata_get_u32(&meta_view, "fireredvad.D", &model->arch.D);
    esp_model_metadata_get_u32(&meta_view, "fireredvad.H", &model->arch.H);
    esp_model_metadata_get_u32(&meta_view, "fireredvad.P", &model->arch.P);
    esp_model_metadata_get_u32(&meta_view, "fireredvad.odim", &model->arch.odim);
    esp_model_metadata_get_u32(&meta_view, "fireredvad.R", &model->arch.R);
    esp_model_metadata_get_u32(&meta_view, "fireredvad.M", &model->arch.M);
    esp_model_metadata_get_u32(&meta_view, "fireredvad.N1", &model->arch.N1);
    esp_model_metadata_get_u32(&meta_view, "fireredvad.S1", &model->arch.S1);
    esp_model_metadata_get_u32(&meta_view, "fireredvad.N2", &model->arch.N2);
    esp_model_metadata_get_u32(&meta_view, "fireredvad.S2", &model->arch.S2);
    esp_model_metadata_get_u32(&meta_view, "fireredvad.cmvn_dim", &model->cmvn_dim);

    const char *mtype_str = NULL;
    size_t mtype_len = 0;
    if (esp_model_metadata_get_utf8(&meta_view, "fireredvad.model_type", &mtype_str, &mtype_len) == ESP_OK) {
        if (strncmp(mtype_str, "stream-vad", mtype_len) == 0) model->model_type = 1;
        else if (strncmp(mtype_str, "aed", mtype_len) == 0) model->model_type = 2;
        else model->model_type = 0;
    }

    // Set quantization properties
    if (container.header.quant_type == MODEL_QUANT_INT8_PER_CH) {
        model->is_int8 = true;
        model->is_int8_per_ch = true;
        model->version = 4;
    } else if (container.header.quant_type == MODEL_QUANT_INT8) {
        model->is_int8 = true;
        model->is_int8_per_ch = false;
        model->version = 2;
    } else {
        model->is_int8 = false;
        model->is_int8_per_ch = false;
        model->version = 1;
    }

    // Validate tensor index
    esp_model_tensor_index_header_v2_t idx_header;
    const esp_model_tensor_record_v2_t *records = NULL;
    err = esp_model_tensor_index_v2_validate(
        raw_data + sec_idx->offset, sec_idx->size, sec_data->size, &idx_header, &records);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Tensor index validation failed (error %d)", err);
        firevad_free_aligned(raw_data);
        return err;
    }

    const uint8_t *tensor_data = raw_data + sec_data->offset;
    const size_t tensor_data_size = (size_t)sec_data->size;

    // CMVN
    if (model->cmvn_dim > 0) {
        const esp_model_tensor_record_v2_t *rec_m = find_tensor(records, idx_header.tensor_count, "cmvn.means");
        const esp_model_tensor_record_v2_t *rec_s = find_tensor(records, idx_header.tensor_count, "cmvn.istd");
        if (rec_m && rec_m->data_offset + rec_m->data_size <= tensor_data_size) {
            model->cmvn_means = (const float *)(tensor_data + rec_m->data_offset);
        }
        if (rec_s && rec_s->data_offset + rec_s->data_size <= tensor_data_size) {
            model->cmvn_istd = (const float *)(tensor_data + rec_s->data_offset);
        }
    }

    const uint32_t D = model->arch.D;
    const uint32_t H = model->arch.H;
    const uint32_t P = model->arch.P;
    const uint32_t R = model->arch.R;
    const uint32_t M = model->arch.M;

    // FC1 & FC2
    bind_dense_layer(&model->fc1,
                     find_tensor(records, idx_header.tensor_count, "fc1.weight"),
                     find_tensor(records, idx_header.tensor_count, "fc1.bias"),
                     tensor_data, tensor_data_size, D, H);

    bind_dense_layer(&model->fc2,
                     find_tensor(records, idx_header.tensor_count, "fc2.weight"),
                     find_tensor(records, idx_header.tensor_count, "fc2.bias"),
                     tensor_data, tensor_data_size, H, P);

    // FSMN 1
    bind_fsmn_filter(&model->fsmn1,
                     find_tensor(records, idx_header.tensor_count, "fsmn1.lookback_weight"),
                     find_tensor(records, idx_header.tensor_count, "fsmn1.lookahead_weight"),
                     tensor_data, tensor_data_size);

    // R - 1 DFSMN blocks
    uint32_t num_blocks = (R > 1) ? R - 1 : 0;
    if (num_blocks > 0) {
        model->block_fc1 = (DenseLayer *)firevad_alloc_aligned(num_blocks * sizeof(DenseLayer), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        model->block_fc2 = (DenseLayer *)firevad_alloc_aligned(num_blocks * sizeof(DenseLayer), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        model->block_fsmn = (FsmnFilter *)firevad_alloc_aligned(num_blocks * sizeof(FsmnFilter), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

        if (!model->block_fc1 || !model->block_fc2 || !model->block_fsmn) {
            ESP_LOGE(TAG, "Failed to allocate block layer arrays");
            esp_firevad_free(model);
            firevad_free_aligned(raw_data);
            return ESP_ERR_NO_MEM;
        }

        char name_buf[64];
        for (uint32_t b = 0; b < num_blocks; ++b) {
            snprintf(name_buf, sizeof(name_buf), "block_fc1.%" PRIu32 ".weight", b);
            const esp_model_tensor_record_v2_t *w1 = find_tensor(records, idx_header.tensor_count, name_buf);
            snprintf(name_buf, sizeof(name_buf), "block_fc1.%" PRIu32 ".bias", b);
            const esp_model_tensor_record_v2_t *b1 = find_tensor(records, idx_header.tensor_count, name_buf);
            bind_dense_layer(&model->block_fc1[b], w1, b1, tensor_data, tensor_data_size, P, H);

            snprintf(name_buf, sizeof(name_buf), "block_fc2.%" PRIu32 ".weight", b);
            const esp_model_tensor_record_v2_t *w2 = find_tensor(records, idx_header.tensor_count, name_buf);
            bind_dense_layer(&model->block_fc2[b], w2, NULL, tensor_data, tensor_data_size, H, P);

            snprintf(name_buf, sizeof(name_buf), "block_fsmn.%" PRIu32 ".lookback_weight", b);
            const esp_model_tensor_record_v2_t *lb = find_tensor(records, idx_header.tensor_count, name_buf);
            snprintf(name_buf, sizeof(name_buf), "block_fsmn.%" PRIu32 ".lookahead_weight", b);
            const esp_model_tensor_record_v2_t *la = find_tensor(records, idx_header.tensor_count, name_buf);
            bind_fsmn_filter(&model->block_fsmn[b], lb, la, tensor_data, tensor_data_size);
        }
    }

    // M DNN layers
    model->num_dnn_layers = M;
    if (M > 0) {
        model->dnn_layers = (DenseLayer *)firevad_alloc_aligned(M * sizeof(DenseLayer), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!model->dnn_layers) {
            ESP_LOGE(TAG, "Failed to allocate dnn_layers array");
            esp_firevad_free(model);
            firevad_free_aligned(raw_data);
            return ESP_ERR_NO_MEM;
        }

        char name_buf[64];
        for (uint32_t d = 0; d < M; ++d) {
            snprintf(name_buf, sizeof(name_buf), "dnn.%" PRIu32 ".weight", d);
            const esp_model_tensor_record_v2_t *w = find_tensor(records, idx_header.tensor_count, name_buf);
            snprintf(name_buf, sizeof(name_buf), "dnn.%" PRIu32 ".bias", d);
            const esp_model_tensor_record_v2_t *b = find_tensor(records, idx_header.tensor_count, name_buf);
            uint32_t in_d = (d == 0) ? P : H;
            bind_dense_layer(&model->dnn_layers[d], w, b, tensor_data, tensor_data_size, in_d, H);
        }
    }

    // Output layer
    bind_dense_layer(&model->out,
                     find_tensor(records, idx_header.tensor_count, "out.weight"),
                     find_tensor(records, idx_header.tensor_count, "out.bias"),
                     tensor_data, tensor_data_size, H, model->arch.odim);

    // Setup streaming caches
    uint32_t lookback_len = (model->arch.N1 > 1) ? (model->arch.N1 - 1) * model->arch.S1 : 0;
    uint32_t lookahead_len = (model->arch.N2 > 0) ? model->arch.N2 * model->arch.S2 : 0;
    model->cache_len = lookback_len + lookahead_len;

    if (model->cache_len > 0) {
        model->fsmn_caches = (float **)firevad_alloc_aligned(R * sizeof(float *), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        model->fsmn_cache_heads = (uint32_t *)firevad_alloc_aligned(R * sizeof(uint32_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!model->fsmn_caches || !model->fsmn_cache_heads) {
            ESP_LOGE(TAG, "Failed to allocate FSMN cache pointers");
            esp_firevad_free(model);
            firevad_free_aligned(raw_data);
            return ESP_ERR_NO_MEM;
        }

        size_t cache_bytes = P * model->cache_len * sizeof(float);
        for (uint32_t r = 0; r < R; ++r) {
            model->fsmn_caches[r] = (float *)firevad_alloc_aligned(cache_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
            if (!model->fsmn_caches[r]) {
                ESP_LOGE(TAG, "Failed to allocate FSMN cache buffer %u", r);
                esp_firevad_free(model);
                firevad_free_aligned(raw_data);
                return ESP_ERR_NO_MEM;
            }
            memset(model->fsmn_caches[r], 0, cache_bytes);
            model->fsmn_cache_heads[r] = 0;
        }
    }

    // Allocate scratch buffers in fast internal memory
    model->scratch_h = (float *)firevad_alloc_aligned(H * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    model->scratch_p = (float *)firevad_alloc_aligned(P * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    model->scratch_p2 = (float *)firevad_alloc_aligned(P * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    model->scratch_conv = (float *)firevad_alloc_aligned(P * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    uint32_t max_dim = (H > P) ? H : P;
    if (D > max_dim) max_dim = D;
    model->scratch_in_q = (int8_t *)firevad_alloc_aligned(max_dim * sizeof(int8_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    if (!model->scratch_h || !model->scratch_p || !model->scratch_p2 || !model->scratch_conv || !model->scratch_in_q) {
        ESP_LOGE(TAG, "Failed to allocate inference scratch buffers");
        esp_firevad_free(model);
        firevad_free_aligned(raw_data);
        return ESP_ERR_NO_MEM;
    }

    model->weight_buffer = raw_data;
    model->weight_buffer_size = file_size;
    model->owns_weight_buffer = true;

    ESP_LOGI(TAG, "ESPM v3 model loaded successfully: type=%u, quant=%u, R=%u, M=%u, cache=%u frames",
             model->model_type, container.header.quant_type, R, M, model->cache_len);
    return ESP_OK;
}
