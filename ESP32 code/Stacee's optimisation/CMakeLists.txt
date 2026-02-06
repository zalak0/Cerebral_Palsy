cmake_minimum_required(VERSION 3.16)

# Add global compile definitions
add_compile_definitions(
    ESP32_ARDUINO_NO_RGB_BUILTIN=1
    FASTLED_RMT5=1
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(baby_cry_detector)