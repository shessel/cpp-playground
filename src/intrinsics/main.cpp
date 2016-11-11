#include <array>
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

vec4f dot4(const std::array<vec4f,4>& a, const std::array<vec4f, 4>& b) {
    __m128 dp = _mm_or_ps(_mm_dp_ps(a[0].data.raw, b[0].data.raw, 0xF1),
                _mm_or_ps(_mm_dp_ps(a[1].data.raw, b[1].data.raw, 0xF2),
                _mm_or_ps(_mm_dp_ps(a[2].data.raw, b[2].data.raw, 0xF4),
                          _mm_dp_ps(a[3].data.raw, b[4].data.raw, 0xF8))));
    return dp;
}

void testDot() {
    std::cout << "dot: " << dot(vec4f{1.0f, 1.0f, 0.0f, 0.0f}, vec4f{1.0f, 1.0f, 0.0f, 0.0f}) << std::endl;
    std::cout << "dot: " << dot(vec4f{2.0f, 3.0f, 0.0f, 0.0f}, vec4f{5.0f, 5.0f, 0.0f, 0.0f}) << std::endl;
}

void testDot4() {
    std::array<vec4f, 4> dotInA{{
        vec4f{1.0f, 1.0f, 1.0f, 1.0f},
        vec4f{1.0f, 1.0f, 1.0f, 1.0f},
        vec4f{1.0f, 1.0f, 1.0f, 1.0f},
        vec4f{1.0f, 1.0f, 1.0f, 1.0f}
    }};
    std::array<vec4f, 4> dotInB{{
        vec4f{1.0f, 0.0f, 0.0f, 0.0f},
        vec4f{1.0f, 1.0f, 0.0f, 0.0f},
        vec4f{1.0f, 1.0f, 1.0f, 0.0f},
        vec4f{1.0f, 1.0f, 1.0f, 1.0f}
    }};
    vec4f dot4Res{dot4(dotInA, dotInB)};

    std::cout << "dot4: ";
    for (double val : dot4Res.data.val) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

void testSetLoadMulAdd() {
    __m256d v1 = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
    __m256d v2 = _mm256_set_pd(2.0, 3.0, 4.0, 5.0);

    double values[4] = {0.0, 1.0, 0.0, 1.0};
    double values2[4] = {0.0, 1.0, 0.0, 1.0};
    __m256d v3 = _mm256_load_pd(values);
    __m256d v4 = _mm256_load_pd(values2);

    vec4d res{_mm256_mul_pd(v1, v2)};
    res.data.raw = _mm256_add_pd(res.data.raw, v3);
    res.data.raw = _mm256_add_pd(res.data.raw, v4);

    std::cout << "SetLoadMulAdd: ";
    for (double val : res.data.val) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    testDot();
    testDot4();
    testSetLoadMulAdd();

    return 0;
}

