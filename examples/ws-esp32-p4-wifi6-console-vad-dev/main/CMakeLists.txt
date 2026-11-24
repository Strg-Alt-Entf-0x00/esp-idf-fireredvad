idf_component_register(
    SRCS "main.cpp" "cmd_vad_cli.cpp" "wifi_transfer.cpp" "audio_wav.c"
    INCLUDE_DIRS "."
    REQUIRES 
        esp-idf-fireredvad
        vad_app
        console
        nvs_flash
        espressif__esp_codec_dev
        fatfs
        sdmmc
        esp_hosted
        esp_wifi_remote
        esp_wifi
        esp_netif
        esp_event
        esp_http_server
        esp_driver_sdmmc
        esp_driver_uart
    PRIV_REQUIRES
        driver
        esp_driver_gpio
        esp_driver_i2c
        esp_driver_i2s
        esp_timer
        esp_pm
        esp_ringbuf
)
