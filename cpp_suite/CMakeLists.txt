cmake_minimum_required(VERSION 3.16)
project(Hydrogen VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_AVX2 "Enable AVX2/FMA/SSE4.1 codegen on x86_64 hosts" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(Threads REQUIRED)

set(CORE_SOURCES
    src/sdr_device.cpp
    src/virtual_sdr.cpp
    src/sdr_controller.cpp
    src/dsp_pipeline.cpp
    src/cs16_writer.cpp
    src/waterfall.cpp
    src/sidecar_reader.cpp
    src/streaming_processor.cpp
    src/dc_excision.cpp
    src/rfi_filter.cpp
    src/baseline_fitter.cpp
    src/astrophysics.cpp
    src/plot_renderer.cpp
    src/exporter.cpp
    lib/kissfft/kiss_fft.c
)

add_library(astro_core STATIC ${CORE_SOURCES})
target_include_directories(astro_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/kissfft
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/rtlsdr
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/stb
)
target_link_libraries(astro_core PUBLIC Threads::Threads)

if (MSVC)
    target_compile_options(astro_core PUBLIC $<$<NOT:$<CONFIG:Debug>>:/O2> /W3 /MP /utf-8)
    if (ENABLE_AVX2 AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64"))
        target_compile_options(astro_core PUBLIC /arch:AVX2)
    endif()
else()
    target_compile_options(astro_core PUBLIC $<$<NOT:$<CONFIG:Debug>>:-O3> -Wall -Wextra -Wno-unused-parameter)
    if (ENABLE_AVX2 AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64"))
        target_compile_options(astro_core PUBLIC -mavx2 -mfma -msse4.1)
    endif()
    if (UNIX AND NOT APPLE)
        target_link_libraries(astro_core PUBLIC ${CMAKE_DL_LIBS})
    endif()
endif()

if (WIN32)
    target_compile_definitions(astro_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
    set(WINDOWS_LIBS
        d3d11 dxgi d3dcompiler dwmapi
        ole32 kernel32 user32 shell32 gdi32 psapi comdlg32
    )
    target_link_libraries(astro_core PUBLIC ole32 kernel32 user32 psapi)

    # 1. Desktop GUI Executable (Windows DX11 + Dear ImGui)
    add_executable(hydrogen
        lib/imgui/imgui.cpp
        lib/imgui/imgui_draw.cpp
        lib/imgui/imgui_widgets.cpp
        lib/imgui/imgui_tables.cpp
        lib/imgui/imgui_impl_win32.cpp
        lib/imgui/imgui_impl_dx11.cpp
        src/ui_record_panel.cpp
        src/ui_analyze_panel.cpp
        src/app_gui.cpp
        src/main_gui.cpp
        assets/hydrogen.rc
    )
    target_include_directories(hydrogen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui)
    target_link_libraries(hydrogen PRIVATE astro_core ${WINDOWS_LIBS})
    if (MINGW)
        target_link_options(hydrogen PRIVATE -static -static-libgcc -static-libstdc++ -mwindows)
    elseif (MSVC)
        set_target_properties(hydrogen PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
    endif()

    add_custom_command(TARGET hydrogen POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/bin/rtlsdr.dll" $<TARGET_FILE_DIR:hydrogen>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/bin/libusb-1.0.dll" $<TARGET_FILE_DIR:hydrogen>
    )
endif()

# 2. Headless CLI Executable
if (WIN32)
    add_executable(hydrogen_cli src/main_cli.cpp assets/hydrogen.rc)
else()
    add_executable(hydrogen_cli src/main_cli.cpp)
endif()
target_link_libraries(hydrogen_cli PRIVATE astro_core)
if (MINGW)
    target_link_options(hydrogen_cli PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# 3. Automated Tests
enable_testing()

function(add_astro_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE astro_core)
    if (MINGW)
        target_link_options(${name} PRIVATE -static -static-libgcc -static-libstdc++)
    endif()
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

add_astro_test(test_suite_state tests/test_suite_state.cpp)
add_astro_test(test_end_to_end_bridge tests/test_end_to_end_bridge.cpp)
add_astro_test(test_cli_commands tests/test_cli_commands.cpp)
add_dependencies(test_cli_commands hydrogen_cli)
add_astro_test(test_performance_stress tests/test_performance_stress.cpp)
add_astro_test(test_dsp_optimizations tests/test_dsp_optimizations.cpp)
