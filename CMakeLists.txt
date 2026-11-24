idf_component_register(
    SRCS
        "src/esp_firevad.cpp"
        "src/esp_firevad_dsp.cpp"
        "src/loaders/firevad_espm_loader_core.c"
        "src/loaders/firevad_espm_loader_vfs.c"
    INCLUDE_DIRS
        "include"
    PRIV_INCLUDE_DIRS
        "src"
        "src/loaders"
    REQUIRES
        esp-dsp
        esp_timer
        esp-idf-model-manager
)

# Aggressive compiler optimization flags for FireRedVAD inference performance
target_compile_options(${COMPONENT_LIB} PRIVATE 
    -Ofast                    # Enable all -O3 optimizations + aggressive FP math
    -funroll-loops            # Force loop unrolling for FSMN/Dense layers
    -fno-strict-aliasing      # Allow aggressive pointer optimization
    -falign-functions=32      # 32-byte function alignment for ESP32-P4 cache (32KB L1)
    -falign-loops=16          # 16-byte loop alignment for better instruction fetch
)

# Keep non-finite ESPM payload validation reliable despite fast-math optimization flags
set_source_files_properties(src/loaders/firevad_espm_loader_core.c PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math"
)
