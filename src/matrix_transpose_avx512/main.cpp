#include <cstdint>

#include <iostream>
#include <memory>

#include <immintrin.h>

#include <benchmark/benchmark.h>

namespace {
[[maybe_unused]]
bool is_aligned(const void* ptr, size_t alignment) {
    // alignment is power of 2
    assert((alignment & (alignment - 1)) == 0);
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

template <typename T, size_t M, size_t N>
struct alignas(64) Matrix {
    T values[M][N];
};

using float8x8 = Matrix<float,8,8>;

struct float8x8simd {
    __m256 rows[8];

    void load(const float * const ptr[8]) {
        for (size_t i = 0; i < 8; ++i) {
            rows[i] = _mm256_load_ps(ptr[i]);
        }
    }

    void load(const float8x8 &matrix) {
        for (size_t i = 0; i < 8; ++i) {
            rows[i] = _mm256_load_ps(matrix.values[i]);
        }
    }

    void store(float *ptr) {
        for (size_t i = 0; i < 8; ++i, ptr+=8) {
            //_mm256_store_ps dst must be 32 byte aligned
            assert(is_aligned(ptr, 32));
            _mm256_store_ps(ptr, rows[i]);
        }
    }

    void store(float *ptr[8]) {
        for (size_t i = 0; i < 8; ++i) {
            //_mm256_store_ps dst must be 32 byte aligned
            assert(is_aligned(ptr[i], 32));
            _mm256_store_ps(ptr[i], rows[i]);
        }
    }

    void store(float8x8& matrix) {
        for (size_t i = 0; i < 8; ++i) {
            //_mm256_store_ps dst must be 32 byte aligned
            assert(is_aligned(matrix.values[i], 32));
            _mm256_store_ps(matrix.values[i], rows[i]);
        }
    }
};

[[maybe_unused]]
void transpose_inplace(float8x8simd &matrix) {

    // interleave elements 0,1 and 4,5 of two rows eg:
    // row0 [00 01 02 03 04 05 06 07]
    // row1 [10 11 12 13 14 15 16 17]
    // to
    // row_0_1_shuffle0101 [00 01 10 11 04 05 14 15]
    const uint8_t shuffle0101 = 0b01'00'01'00;
    __m256 row_0_1_shuffle0101 = _mm256_shuffle_ps(matrix.rows[0], matrix.rows[1], shuffle0101);
    __m256 row_2_3_shuffle0101 = _mm256_shuffle_ps(matrix.rows[2], matrix.rows[3], shuffle0101);
    __m256 row_4_5_shuffle0101 = _mm256_shuffle_ps(matrix.rows[4], matrix.rows[5], shuffle0101);
    __m256 row_6_7_shuffle0101 = _mm256_shuffle_ps(matrix.rows[6], matrix.rows[7], shuffle0101);


    // interleave elements 2,3 and 6,7 of two rows eg:
    // row0 [00 01 02 03 04 05 06 07]
    // row1 [10 11 12 13 14 15 16 17]
    // to
    // row_0_1_shuffle2323 [02 03 12 13 06 07 16 17]
    const uint8_t shuffle2323 = 0b11'10'11'10;
    __m256 row_0_1_shuffle2323 = _mm256_shuffle_ps(matrix.rows[0], matrix.rows[1], shuffle2323);
    __m256 row_2_3_shuffle2323 = _mm256_shuffle_ps(matrix.rows[2], matrix.rows[3], shuffle2323);
    __m256 row_4_5_shuffle2323 = _mm256_shuffle_ps(matrix.rows[4], matrix.rows[5], shuffle2323);
    __m256 row_6_7_shuffle2323 = _mm256_shuffle_ps(matrix.rows[6], matrix.rows[7], shuffle2323);

    // interleave elements 0, 2 and 4, 6 of two shuffeled rows eg:
    // row_0_1_shuffle0101 [00 01 10 11 04 05 14 15]
    // row_2_3_shuffle0101 [20 21 30 31 24 25 34 35]
    // to
    // row_0_1_2_3_shuffle_0101_0202 [00 10 20 30 04 14 24 34]
    // and
    // row_0_1_shuffle2323 [02 03 12 13 06 07 16 17]
    // row_2_3_shuffle2323 [22 23 32 33 26 27 36 37]
    // to
    // row_0_1_2_3_shuffle_2323_0202 [02 12 22 32 06 16 26 36]
    const uint8_t shuffle0202 = 0b10'00'10'00;
    __m256 row_0_1_2_3_shuffle_0101_0202 = _mm256_shuffle_ps(row_0_1_shuffle0101, row_2_3_shuffle0101, shuffle0202);
    __m256 row_4_5_6_7_shuffle_0101_0202 = _mm256_shuffle_ps(row_4_5_shuffle0101, row_6_7_shuffle0101, shuffle0202);
    __m256 row_0_1_2_3_shuffle_2323_0202 = _mm256_shuffle_ps(row_0_1_shuffle2323, row_2_3_shuffle2323, shuffle0202);
    __m256 row_4_5_6_7_shuffle_2323_0202 = _mm256_shuffle_ps(row_4_5_shuffle2323, row_6_7_shuffle2323, shuffle0202);

    // interleave elements 1, 3 and 5, 7 of two shuffeled rows eg:
    // row_0_1_shuffle0101 [00 01 10 11 04 05 14 15]
    // row_2_3_shuffle0101 [20 21 30 31 24 25 34 35]
    // to
    // row_0_1_2_3_shuffle_0101_1313 [01 11 21 31 05 15 25 35]
    // and
    // row_0_1_shuffle2323 [02 03 12 13 06 07 16 17]
    // row_2_3_shuffle2323 [22 23 32 33 26 27 36 37]
    // to
    // row_0_1_2_3_shuffle_2323_1313 [03 13 23 33 07 17 27 37]
    const uint8_t shuffle1313 = 0b11'01'11'01;
    __m256 row_0_1_2_3_shuffle_0101_1313 = _mm256_shuffle_ps(row_0_1_shuffle0101, row_2_3_shuffle0101, shuffle1313);
    __m256 row_4_5_6_7_shuffle_0101_1313 = _mm256_shuffle_ps(row_4_5_shuffle0101, row_6_7_shuffle0101, shuffle1313);
    __m256 row_0_1_2_3_shuffle_2323_1313 = _mm256_shuffle_ps(row_0_1_shuffle2323, row_2_3_shuffle2323, shuffle1313);
    __m256 row_4_5_6_7_shuffle_2323_1313 = _mm256_shuffle_ps(row_4_5_shuffle2323, row_6_7_shuffle2323, shuffle1313);

    // join low 4 elements from two twice-shuffeled rows:
    // row_0_1_2_3_shuffle_0101_0202 [00 10 20 30 04 14 24 34]
    // row_4_5_6_7_shuffle_0101_0202 [40 50 60 70 44 54 64 74]
    // to
    // row0 = [00 10 20 30 40 50 60 70]
    const uint8_t shuffle_low4 = 0b0'0;
    matrix.rows[0] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_0101_0202, row_4_5_6_7_shuffle_0101_0202, shuffle_low4);
    matrix.rows[1] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_0101_1313, row_4_5_6_7_shuffle_0101_1313, shuffle_low4);
    matrix.rows[2] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_2323_0202, row_4_5_6_7_shuffle_2323_0202, shuffle_low4);
    matrix.rows[3] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_2323_1313, row_4_5_6_7_shuffle_2323_1313, shuffle_low4);

    // join high 4 elements from two twice-shuffeled rows:
    // row_0_1_2_3_shuffle_0101_0202 [00 10 20 30 04 14 24 34]
    // row_4_5_6_7_shuffle_0101_0202 [40 50 60 70 44 54 64 74]
    // to
    // row4 = [04 14 24 34 44 54 64 74]
    const uint8_t shuffle_high4 = 0b1'1;
    matrix.rows[4] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_0101_0202, row_4_5_6_7_shuffle_0101_0202, shuffle_high4);
    matrix.rows[5] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_0101_1313, row_4_5_6_7_shuffle_0101_1313, shuffle_high4);
    matrix.rows[6] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_2323_0202, row_4_5_6_7_shuffle_2323_0202, shuffle_high4);
    matrix.rows[7] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_2323_1313, row_4_5_6_7_shuffle_2323_1313, shuffle_high4);
}

[[maybe_unused]]
void transpose_inplace(float8x8& matrix) {
    float8x8simd matrix_simd;
    matrix_simd.load(matrix);
    transpose_inplace(matrix_simd);
    matrix_simd.store(matrix);
}

[[maybe_unused]]
void transpose_inplace(Matrix<float, 8 * 64, 8 * 64>& matrix) {
    // transpose 8x8 submatrices
    for (size_t row = 0; row < 8*(64-1); row += 8) {
        for (size_t col = row + 8; col < 8 * 64; col += 8) {
            float* row_submatrix_rows[8];
            for (size_t i = 0; i < 8; ++i) {
                row_submatrix_rows[i] = &matrix.values[row + i][col];
            }
            float8x8simd row_submatrix;
            row_submatrix.load(row_submatrix_rows);
            transpose_inplace(row_submatrix);
            row_submatrix.store(row_submatrix_rows);

            float* col_submatrix_rows[8];
            for (size_t i = 0; i < 8; ++i) {
                col_submatrix_rows[i] = &matrix.values[col + i][row];
            }
            float8x8simd col_submatrix;
            col_submatrix.load(col_submatrix_rows);
            transpose_inplace(col_submatrix);

            // store in transposed positions
            row_submatrix.store(col_submatrix_rows);
            col_submatrix.store(row_submatrix_rows);
        }
    }
    // handle diagonal submatrices
    for (size_t row_col = 0; row_col < 8 * 64; row_col += 8) {
        float* submatrix_rows[8];
        for (size_t i = 0; i < 8; ++i) {
            submatrix_rows[i] = &matrix.values[row_col + i][row_col];
        }
        float8x8simd submatrix;
        submatrix.load(submatrix_rows);
        transpose_inplace(submatrix);
        submatrix.store(submatrix_rows);
    }
}

[[maybe_unused]]
void transpose_inplace_swap(float8x8& matrix) {
    for (size_t y = 0; y < 7; ++y) {
        for (size_t x = y + 1; x < 8; ++x) {
            std::swap(matrix.values[y][x], matrix.values[x][y]);
        }
    }
}

template <typename T, size_t N>
[[maybe_unused]]
void transpose_inplace_swap_square(Matrix<T,N,N>& matrix) {
    for (size_t row = 0; row < N; ++row) {
        for (size_t col = row + 1; col < N; ++col) {
            std::swap(matrix.values[row][col], matrix.values[col][row]);
        }
    }
}

template <typename T, size_t N>
[[maybe_unused]]
void transpose_inplace_swap_square_blocked(Matrix<T, N, N>& matrix) {
    const size_t BLOCK_SIZE = 8;
    for (size_t row = 0; row < N; row+=BLOCK_SIZE) {
        for (size_t col = row + BLOCK_SIZE; col < N; col+=BLOCK_SIZE) {
            for (size_t submatrix_row = row; submatrix_row < row+BLOCK_SIZE; ++submatrix_row) {
                for (size_t submatrix_col = col; submatrix_col < col + BLOCK_SIZE; ++submatrix_col) {
                    std::swap(matrix.values[submatrix_row][submatrix_col], matrix.values[submatrix_col][submatrix_row]);
                }
            }
        }
    }

    for (size_t row_col = 0; row_col < N; row_col += BLOCK_SIZE) {
        for (size_t submatrix_row = row_col; submatrix_row < row_col + BLOCK_SIZE - 1; ++submatrix_row) {
            for (size_t submatrix_col = submatrix_row + 1; submatrix_col < row_col + BLOCK_SIZE; ++submatrix_col) {
                std::swap(matrix.values[submatrix_row][submatrix_col], matrix.values[submatrix_col][submatrix_row]);
            }
        }
    }
}

[[maybe_unused]]
float8x8 transpose(const float8x8& matrix) {
    float8x8 transposed = matrix;
    transpose_inplace(transposed);
    return transposed;
}
/*
static void BM_TransposeAVX(benchmark::State& state) {
    float8x8 test = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f,
        20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f,
        30.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 37.f,
        40.f, 41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f,
        50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f, 57.f,
        60.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 67.f,
        70.f, 71.f, 72.f, 73.f, 74.f, 75.f, 76.f, 77.f,
    };
    for (auto _ : state)
        transpose_inplace(test);
}
BENCHMARK(BM_TransposeAVX);

static void BM_TransposeSwap(benchmark::State& state) {
    float8x8 test = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f,
        20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f,
        30.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 37.f,
        40.f, 41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f,
        50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f, 57.f,
        60.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 67.f,
        70.f, 71.f, 72.f, 73.f, 74.f, 75.f, 76.f, 77.f,
    };
    for (auto _ : state)
        transpose_inplace_swap(test);
}
BENCHMARK(BM_TransposeSwap);
*/
[[maybe_unused]]
static void VM_TransposeAVXBig(benchmark::State& state) {
    auto big_test = std::make_unique<Matrix<float, 8 * 64, 8 * 64>>();
    for (size_t row = 0; row < 8 * 64; ++row) {
        for (size_t col = 0; col < 8 * 64; ++col) {
            big_test->values[row][col] = static_cast<float>(row) * 1.0f + static_cast<float>(col) * 0.001f;
        }
    }
    for (auto _ : state)
        transpose_inplace(*big_test);
}

[[maybe_unused]]
static void VM_TransposeSwapBig(benchmark::State& state) {
    auto big_test = std::make_unique<Matrix<float, 8 * 64, 8 * 64>>();
    for (size_t row = 0; row < 8 * 64; ++row) {
        for (size_t col = 0; col < 8 * 64; ++col) {
            big_test->values[row][col] = static_cast<float>(row) * 1.0f + static_cast<float>(col) * 0.001f;
        }
    }
    for (auto _ : state)
        transpose_inplace_swap_square(*big_test);
}

[[maybe_unused]]
static void VM_TransposeSwapBlockedBig(benchmark::State& state) {
    auto big_test = std::make_unique<Matrix<float, 8 * 64, 8 * 64>>();
    for (size_t row = 0; row < 8 * 64; ++row) {
        for (size_t col = 0; col < 8 * 64; ++col) {
            big_test->values[row][col] = static_cast<float>(row) * 1.0f + static_cast<float>(col) * 0.001f;
        }
    }
    for (auto _ : state)
        transpose_inplace_swap_square_blocked(*big_test);
}
}
BENCHMARK(VM_TransposeSwapBlockedBig);
BENCHMARK(VM_TransposeSwapBig);
BENCHMARK(VM_TransposeAVXBig);


#if 1
BENCHMARK_MAIN();
#else
int main() {
    float8x8 test = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f,
        20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f,
        30.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 37.f,
        40.f, 41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f,
        50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f, 57.f,
        60.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 67.f,
        70.f, 71.f, 72.f, 73.f, 74.f, 75.f, 76.f, 77.f,
    };

    for (size_t y = 0; y < 8; ++y) {
        for (size_t x = 0; x < 8; ++x) {
            std::cout << test.values[y][x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    transpose_inplace(test);
    for (size_t y = 0; y < 8; ++y) {
        for (size_t x = 0; x < 8; ++x) {
            std::cout << test.values[y][x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    float8x8 test2 = transpose(test);
    for (size_t y = 0; y < 8; ++y) {
        for (size_t x = 0; x < 8; ++x) {
            std::cout << test2.values[y][x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    transpose_inplace_swap(test2);
    for (size_t y = 0; y < 8; ++y) {
        for (size_t x = 0; x < 8; ++x) {
            std::cout << test2.values[y][x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    auto big_test = std::make_unique<Matrix<float, 8 * 64, 8 * 64>>();
    auto big_test2 = std::make_unique<Matrix<float, 8 * 64, 8 * 64>>();
    auto big_test3 = std::make_unique<Matrix<float, 8 * 64, 8 * 64>>();
    for (size_t row = 0; row < 8 * 64; ++row) {
        for (size_t col = 0; col < 8 * 64; ++col) {
            big_test->values[row][col] = row * 1.0f + col * 0.001f;
            big_test2->values[row][col] = row * 1.0f + col * 0.001f;
            big_test3->values[row][col] = row * 1.0f + col * 0.001f;
        }
    }

    for (size_t row = 196; row < 204; ++row) {
        for (size_t col = 64; col < 72; ++col) {
            std::cout << big_test->values[row][col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    transpose_inplace(*big_test);
    transpose_inplace_swap_square(*big_test2);
    transpose_inplace_swap_square_blocked(*big_test3);

    for (size_t row = 196; row < 204; ++row) {
        for (size_t col = 64; col < 72; ++col) {
            std::cout << big_test->values[row][col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    bool all_equal = true;
    for (size_t row = 0; row < 8 * 64; ++row) {
        for (size_t col = 0; col < 8 * 64; ++col) {
            if (big_test->values[row][col] != big_test2->values[row][col]) {
                std::cout << "AVX transpose differs from swap at row " << row << " col " << col << ": " << big_test->values[row][col] << " != " << big_test2->values[row][col] << "\n";
                all_equal = false;
            }
            if (big_test2->values[row][col] != big_test3->values[row][col]) {
                std::cout << "swap transpose differs from swap blocked at row " << row << " col " << col << ": " << big_test2->values[row][col] << " != " << big_test3->values[row][col] << "\n";
                all_equal = false;
            }
        }
    }
    if (all_equal) {
        std::cout << "All elements in bigtest matrices are equal\n";
    }

    return 0;
}
#endif
