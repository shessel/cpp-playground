cmake_minimum_required(VERSION 3.28.0 FATAL_ERROR)

set(EXE_NAME atomics)
add_executable(${EXE_NAME} main.cpp)

# compiler warnings
# adapted from https://arne-mertz.de/2018/07/cmake-properties-options/
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # warning level 4 and all warnings as errors
    target_compile_options(${EXE_NAME} PRIVATE /W4 /WX /permissive- /arch:AVX512)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CLANG_COMPILE_OPTIONS)
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        list(APPEND CLANG_COMPILE_OPTIONS
            /clang:-Wall
            -fcolor-diagnostics
            -fansi-escape-codes
        )
        # lots of warnings and all warnings as errors
    else()
        list(APPEND CLANG_COMPILE_OPTIONS
            -Wall
            -fdiagnostics-color=always
        )
    endif()
        list(APPEND CLANG_COMPILE_OPTIONS 
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wmissing-prototypes
            -Wno-c++98-compat-pedantic
            -Wno-pre-c++20-compat-pedantic
            -Wno-missing-field-initializers
            -Werror
        )
        # lots of warnings and all warnings as errors
        target_compile_options(${EXE_NAME} PRIVATE ${CLANG_COMPILE_OPTIONS})
endif()

target_compile_features(${EXE_NAME} PRIVATE cxx_std_23)
