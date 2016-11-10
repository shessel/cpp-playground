#include <iostream>
#include "immintrin.h"
#include "smmintrin.h"

struct vec4d {
    union {
        __m256d raw;
        double val[4];
    } data;

    vec4d(double x, double y, double z, double w) : data{x, y, z, w} {}
    vec4d(__m256d raw) : data{raw} {}
};

struct vec4f {
    union {
        __m128 raw;
        float val[4];
    } data;

    vec4f(float x, float y, float z, float w) : data{x, y, z, w} {}
    vec4f(__m128 raw) : data{raw} {}
};

float dot(const vec4f& a, const vec4f& b) {
    __m128 dp = _mm_dp_ps(a.data.raw, b.data.raw, 0xF1);
    return _mm_cvtss_f32(dp);
}

int main() {
    std::cout << dot(vec4f{1.0f, 1.0f, 0.0f, 0.0f}, vec4f{1.0f, 1.0f, 0.0f, 0.0f}) << std::endl;
    std::cout << dot(vec4f{2.0f, 3.0f, 0.0f, 0.0f}, vec4f{5.0f, 5.0f, 0.0f, 0.0f}) << std::endl;

    __m256d v1 = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
    __m256d v2 = _mm256_set_pd(2.0, 3.0, 4.0, 5.0);

    double values[4] = {0.0, 1.0, 0.0, 1.0};
    double values2[4] = {0.0, 1.0, 0.0, 1.0};
    __m256d v3 = _mm256_load_pd(values);
    __m256d v4 = _mm256_load_pd(values2);

    vec4d res{_mm256_mul_pd(v1, v2)};
    res.data.raw = _mm256_add_pd(res.data.raw, v3);
    res.data.raw = _mm256_add_pd(res.data.raw, v4);

    for (double val : res.data.val) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

