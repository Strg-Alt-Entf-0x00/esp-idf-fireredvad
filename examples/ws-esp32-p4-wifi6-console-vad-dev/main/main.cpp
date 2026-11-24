/**
 * @file main.cpp
 * @brief FireVAD Console Example - Professional Voice Activity Detection
 */

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <sys/stat.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_console.h"
#include "nvs_flash.h"
#include "esp_chip_info.h"
#include "esp_pm.h"
#include "esp_idf_version.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"
#include "soc/soc_caps.h"
#if defined(CONFIG_IDF_TARGET_ESP32P4)
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#endif

#include "esp_firevad.h"
#include "esp_firevad_dsp.h"
#include "config.h"
#include "cmd_vad_cli.h"
#include "audio_manager.h"
#include "vad_runner.h"
#include "wifi_transfer.h"

static const char* TAG = "MAIN";

#include "esp_heap_caps.h"

static void print_banner(void) {
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    
    size_t psram_size = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t sram_size = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    
    int cpu_freq_mhz = 0;
#ifdef CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ
    cpu_freq_mhz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
#endif
    
    printf("\n");
    printf("  FIREVAD CONSOLE\n");
    printf("  ESP32-P4 Edition\n");
    printf("  --------------------------------------------------\n");
    if (cpu_freq_mhz > 0) {
        printf("  Hardware : ESP32-P4 rev %d.%d (%d-Core @ %dMHz)\n", 
               chip_info.revision / 100, chip_info.revision % 100, chip_info.cores, cpu_freq_mhz);
    } else {
        printf("  Hardware : ESP32-P4 rev %d.%d (%d-Core)\n", 
               chip_info.revision / 100, chip_info.revision % 100, chip_info.cores);
    }
    printf("  Memory   : %zu MB PSRAM / %zu KB SRAM\n", 
           psram_size / (1024 * 1024), sram_size / 1024);
    printf("  Version  : v1.1.0\n\n");
}


#if defined(CONFIG_IDF_TARGET_ESP32P4) && \
    defined(CONFIG_ESP_HOSTED_HOST_TRANSPORT_BUS_SDIO) && \
    ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
static esp_err_t shared_sdmmc_host_init(void) { return ESP_OK; }
static esp_err_t shared_sdmmc_host_deinit(void) { return ESP_OK; }
#endif

static bool mount_sd_card(void)
{
#if SOC_SDMMC_HOST_SUPPORTED
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;
#if defined(CONFIG_IDF_TARGET_ESP32P4)
    host.slot = SDMMC_HOST_SLOT_0;
#endif
#if defined(CONFIG_IDF_TARGET_ESP32P4) && \
    defined(CONFIG_ESP_HOSTED_HOST_TRANSPORT_BUS_SDIO) && \
    ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
    host.init = shared_sdmmc_host_init;
    host.deinit = shared_sdmmc_host_deinit;
#endif

#if defined(CONFIG_IDF_TARGET_ESP32P4)
    sd_pwr_ctrl_ldo_config_t ldo_config = {.ldo_chan_id = 4};
    sd_pwr_ctrl_handle_t power_control = nullptr;
    const esp_err_t power_error =
        sd_pwr_ctrl_new_on_chip_ldo(&ldo_config, &power_control);
    if (power_error != ESP_OK) {
        ESP_LOGE(TAG, "Could not enable SD-card power: %s",
                 esp_err_to_name(power_error));
        return false;
    }
    host.pwr_ctrl_handle = power_control;
#endif

    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
#if defined(CONFIG_IDF_TARGET_ESP32P4)
    slot.clk = static_cast<gpio_num_t>(43);
    slot.cmd = static_cast<gpio_num_t>(44);
    slot.d0 = static_cast<gpio_num_t>(39);
    slot.d1 = static_cast<gpio_num_t>(40);
    slot.d2 = static_cast<gpio_num_t>(41);
    slot.d3 = static_cast<gpio_num_t>(42);
    slot.width = 4;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
#endif

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {};
    mount_config.format_if_mount_failed = false;
    mount_config.max_files = 10;
    mount_config.allocation_unit_size = 16 * 1024;
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
    mount_config.disk_status_check_enable = true;
    mount_config.use_one_fat = false;
#endif

    sdmmc_card_t* card = nullptr;
    const esp_err_t error = esp_vfs_fat_sdmmc_mount(
        FS_MOUNT_POINT, &host, &slot, &mount_config, &card);
    if (error != ESP_OK) {
        ESP_LOGW(TAG, "SD-card mount failed: %s", esp_err_to_name(error));
#if defined(CONFIG_IDF_TARGET_ESP32P4)
        sd_pwr_ctrl_del_on_chip_ldo(power_control);
#endif
        return false;
    }

    sdmmc_card_print_info(stdout, card);
    const char *directories[] = {"/sd/models", "/sd/audio", "/sd/results"};
    for (const char *directory : directories) {
        if (mkdir(directory, 0775) != 0 && errno != EEXIST) {
            ESP_LOGW(TAG, "Cannot create %s: errno=%d", directory, errno);
        }
    }
    ESP_LOGI(TAG, "SD card mounted at %s", FS_MOUNT_POINT);
    return true;
#else
    ESP_LOGE(TAG, "SDMMC is not supported by this target");
    return false;
#endif
}

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Starting FireVAD System");
    
#if CONFIG_PM_ENABLE
    esp_pm_config_t pm_config = {
        .max_freq_mhz = 360,
        .min_freq_mhz = 40,
        .light_sleep_enable = true
    };
    ESP_ERROR_CHECK(esp_pm_configure(&pm_config));
    ESP_LOGI(TAG, "Power Management Configured (max 360MHz, light sleep enabled)");
#endif
    
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    if (!mount_sd_card()) {
        ESP_LOGW(TAG, "SD card unavailable; SD-backed model and audio commands will fail");
    }
    const esp_err_t wifi_error = wifi_transfer_init();
    if (wifi_error != ESP_OK) {
        ESP_LOGW(TAG, "WLAN transfer initialization failed: %s", esp_err_to_name(wifi_error));
    }
    esp_firevad_dsp_init();
    audio_manager_init();
    
    // Auto-calibrate background noise for 1 second
    ESP_LOGI(TAG, "===============================================");
    ESP_LOGI(TAG, " PLEASE BE SILENT - CALIBRATING MICROPHONE...  ");
    ESP_LOGI(TAG, "===============================================");
    vad_runner_calibrate_noise_blocking(1);
    vad_runner_set_pre_vad_threshold(0.0f); // Disable Pre-VAD by default for pure NN output
    
    // Play startup chime AFTER calibration to signify system is ready and avoid interfering with the mic
    audio_manager_play_boot_sequence();
    
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_config.prompt = CONSOLE_PROMPT;
    repl_config.max_cmdline_length = CONSOLE_MAX_COMMAND_LINE;
    repl_config.task_stack_size = 8192;  // Default 4096 is too small for VAD inference chain
    // Register VAD and WiFi console commands
    cmd_vad_cli_register();
    wifi_transfer_register_commands();
    
    // Start console REPL
    esp_console_dev_uart_config_t hw_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&hw_config, &repl_config, &repl));
    
    print_banner();
    
    printf("  Type 'help'  to see all commands.\n\n");
    
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}
