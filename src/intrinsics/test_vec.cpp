#include <array>
#include <iostream>
#include "immintrin.h"
#include "smmintrin.h"
#include "vec4.h"

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

    std::cout << "dot4: " << dot4Res << std::endl;
}

void testSetLoadMulAdd() {
    __m256d v1 = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
    __m256d v2 = _mm256_set_pd(2.0, 3.0, 4.0, 5.0);

    double values[4] __attribute__((aligned(32))) = {0.0, 1.0, 0.0, 1.0};
    double values2[4] __attribute__((aligned(32))) = {0.0, 1.0, 0.0, 1.0};
    __m256d v3 = _mm256_load_pd(values);
    __m256d v4 = _mm256_load_pd(values2);

    vec4d res{_mm256_mul_pd(v1, v2)};
    res.data.raw = _mm256_add_pd(res.data.raw, v3);
    res.data.raw = _mm256_add_pd(res.data.raw, v4);

    std::cout << "SetLoadMulAdd: " << res << std::endl;
}

int main() {
    testDot();
    testDot4();
    testSetLoadMulAdd();

    return 0;
}
