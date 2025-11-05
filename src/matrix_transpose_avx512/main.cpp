#include <immintrin.h>
#include <iostream>

#include <benchmark/benchmark.h>

template <typename T, size_t M, size_t N>
struct alignas(32) Matrix {
    T values[M][N];
};

using float8x8 = Matrix<float,8,8>;

void transpose_inplace(float8x8 &matrix) {
    __m256 rows[8];
    for (size_t i = 0; i < 8; ++i) {
        rows[i] = _mm256_load_ps(&matrix.values[i][0]);
    }

    const uint8_t shuffle0101 = 0b01'00'01'00;
    __m256 row_0_1_shuffle0101 = _mm256_shuffle_ps(rows[0], rows[1], shuffle0101);
    __m256 row_2_3_shuffle0101 = _mm256_shuffle_ps(rows[2], rows[3], shuffle0101);
    __m256 row_4_5_shuffle0101 = _mm256_shuffle_ps(rows[4], rows[5], shuffle0101);
    __m256 row_6_7_shuffle0101 = _mm256_shuffle_ps(rows[6], rows[7], shuffle0101);

    const uint8_t shuffle2323 = 0b11'10'11'10;
    __m256 row_0_1_shuffle2323 = _mm256_shuffle_ps(rows[0], rows[1], shuffle2323);
    __m256 row_2_3_shuffle2323 = _mm256_shuffle_ps(rows[2], rows[3], shuffle2323);
    __m256 row_4_5_shuffle2323 = _mm256_shuffle_ps(rows[4], rows[5], shuffle2323);
    __m256 row_6_7_shuffle2323 = _mm256_shuffle_ps(rows[6], rows[7], shuffle2323);

    const uint8_t shuffle0202 = 0b10'00'10'00;
    __m256 row_0_1_2_3_shuffle_0101_0202 = _mm256_shuffle_ps(row_0_1_shuffle0101, row_2_3_shuffle0101, shuffle0202);
    __m256 row_4_5_6_7_shuffle_0101_0202 = _mm256_shuffle_ps(row_4_5_shuffle0101, row_6_7_shuffle0101, shuffle0202);
    __m256 row_0_1_2_3_shuffle_2323_0202 = _mm256_shuffle_ps(row_0_1_shuffle2323, row_2_3_shuffle2323, shuffle0202);
    __m256 row_4_5_6_7_shuffle_2323_0202 = _mm256_shuffle_ps(row_4_5_shuffle2323, row_6_7_shuffle2323, shuffle0202);

    const uint8_t shuffle1313 = 0b11'01'11'01;
    __m256 row_0_1_2_3_shuffle_0101_1313 = _mm256_shuffle_ps(row_0_1_shuffle0101, row_2_3_shuffle0101, shuffle1313);
    __m256 row_4_5_6_7_shuffle_0101_1313 = _mm256_shuffle_ps(row_4_5_shuffle0101, row_6_7_shuffle0101, shuffle1313);
    __m256 row_0_1_2_3_shuffle_2323_1313 = _mm256_shuffle_ps(row_0_1_shuffle2323, row_2_3_shuffle2323, shuffle1313);
    __m256 row_4_5_6_7_shuffle_2323_1313 = _mm256_shuffle_ps(row_4_5_shuffle2323, row_6_7_shuffle2323, shuffle1313);


    const uint8_t shuffle_low4 = 0b0'0;
    rows[0] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_0101_0202, row_4_5_6_7_shuffle_0101_0202, shuffle_low4);
    rows[1] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_0101_1313, row_4_5_6_7_shuffle_0101_1313, shuffle_low4);
    rows[2] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_2323_0202, row_4_5_6_7_shuffle_2323_0202, shuffle_low4);
    rows[3] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_2323_1313, row_4_5_6_7_shuffle_2323_1313, shuffle_low4);

    const uint8_t shuffle_high4 = 0b1'1;
    rows[4] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_0101_0202, row_4_5_6_7_shuffle_0101_0202, shuffle_high4);
    rows[5] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_0101_1313, row_4_5_6_7_shuffle_0101_1313, shuffle_high4);
    rows[6] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_2323_0202, row_4_5_6_7_shuffle_2323_0202, shuffle_high4);
    rows[7] = _mm256_shuffle_f32x4(row_0_1_2_3_shuffle_2323_1313, row_4_5_6_7_shuffle_2323_1313, shuffle_high4);
    
    for (size_t i = 0; i < 8; ++i) {
        _mm256_store_ps(&matrix.values[i][0], rows[i]);
    }
}

void transpose_inplace_swap(float8x8& matrix) {
    for (size_t y = 0; y < 7; ++y) {
        for (size_t x = y + 1; x < 8; ++x) {
            std::swap(matrix.values[y][x], matrix.values[x][y]);
        }
    }
}

float8x8 transpose(const float8x8& matrix) {
    float8x8 transposed = matrix;
    transpose_inplace(transposed);
    return transposed;
}

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

#if 0
BENCHMARK_MAIN();
#else
int main() {
#endif
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

    return 0;
}