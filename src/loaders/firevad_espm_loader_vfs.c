/**
 * @file firevad_espm_loader_vfs.c
 * @brief VFS filesystem wrapper for loading FireRedVAD ESPM models.
 */

#include "firevad_espm_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "esp_log.h"
#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#endif

static const char *TAG = "firevad_vfs";

esp_err_t firevad_load_model_espm_file(EspFirevadModel *model, const char *espm_path)
{
    if (!model || !espm_path) {
        return ESP_ERR_INVALID_ARG;
    }

    struct stat file_stat;
    if (stat(espm_path, &file_stat) != 0) {
        ESP_LOGE(TAG, "Cannot stat ESPM model: %s", espm_path);
        return ESP_ERR_NOT_FOUND;
    }

    if (file_stat.st_size < 128 || (uint64_t)file_stat.st_size > FIREVAD_ESPM_MAX_FILE_SIZE) {
        ESP_LOGE(TAG, "Invalid ESPM file size: %ld bytes", (long)file_stat.st_size);
        return ESP_ERR_INVALID_SIZE;
    }

    const size_t file_size = (size_t)file_stat.st_size;
    FILE *file = fopen(espm_path, "rb");
    if (!file) {
        ESP_LOGE(TAG, "Cannot open ESPM model: %s", espm_path);
        return ESP_ERR_NOT_FOUND;
    }

    uint8_t *raw_data = NULL;
#ifdef ESP_PLATFORM
    raw_data = (uint8_t *)heap_caps_malloc(file_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!raw_data) {
        raw_data = (uint8_t *)heap_caps_malloc(file_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
#else
    raw_data = (uint8_t *)malloc(file_size);
#endif

    if (!raw_data) {
        ESP_LOGE(TAG, "Failed to allocate %zu bytes for ESPM buffer", file_size);
        fclose(file);
        return ESP_ERR_NO_MEM;
    }

    const size_t read_bytes = fread(raw_data, 1, file_size, file);
    const int close_res = fclose(file);
    if (read_bytes != file_size || close_res != 0) {
        ESP_LOGE(TAG, "Failed to read complete ESPM file: read %zu of %zu bytes", read_bytes, file_size);
#ifdef ESP_PLATFORM
        heap_caps_free(raw_data);
#else
        free(raw_data);
#endif
        return ESP_FAIL;
    }

    return firevad_load_model_espm_buffer(model, raw_data, file_size);
}
