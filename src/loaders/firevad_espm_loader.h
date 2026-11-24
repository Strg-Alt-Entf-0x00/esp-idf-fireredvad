/**
 * @file firevad_espm_loader.h
 * @brief ESPM v3 container loader for FireRedVAD DFSMN models.
 */

#pragma once

#include "esp_firevad.h"
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#define FIREVAD_ESPM_MAX_FILE_SIZE (16U * 1024U * 1024U)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Loads a FireRedVAD model from an in-memory ESPM v3 container buffer.
 *        Takes ownership of raw_data on success, or frees it on failure.
 *
 * @param model Target EspFirevadModel struct to populate.
 * @param raw_data Pointer to buffer containing complete .espm binary.
 * @param file_size Buffer size in bytes.
 * @return esp_err_t ESP_OK on success, error code otherwise.
 */
esp_err_t firevad_load_model_espm_buffer(EspFirevadModel *model,
                                        uint8_t *raw_data,
                                        size_t file_size);

/**
 * @brief Loads a FireRedVAD model from an ESPM v3 file on a mounted VFS filesystem.
 *
 * @param model Target EspFirevadModel struct to populate.
 * @param espm_path Absolute path to the .espm model file.
 * @return esp_err_t ESP_OK on success, error code otherwise.
 */
esp_err_t firevad_load_model_espm_file(EspFirevadModel *model,
                                      const char *espm_path);

#ifdef __cplusplus
}
#endif
