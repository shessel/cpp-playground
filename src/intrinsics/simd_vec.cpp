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

std::ostream& operator<<(std::ostream& os, const vec4f& vec) {
    std::string delimiter = "";
    os << "(";
    for (const auto& val : vec.data.val) {
        os << delimiter << val;
        delimiter = ", ";
    }
    os << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const vec4d& vec) {
    std::string delimiter = "";
    os << "(";
    for (const auto& val : vec.data.val) {
        os << delimiter << val;
        delimiter = ", ";
    }
    os << ")";
    return os;
}

struct mat4f {
    __m128 row0;
    __m128 row1;
    __m128 row2;
    __m128 row3;

    explicit mat4f(__m128 row0, __m128 row1, __m128 row2, __m128 row3) :
        row0(row0), row1(row1), row2(row2), row3(row3) {}
    explicit mat4f(vec4f row0, vec4f row1, vec4f row2, vec4f row3) :
        row0(row0.data.raw), row1(row1.data.raw), row2(row2.data.raw), row3(row3.data.raw) {}
    vec4f operator*(const vec4f& vec) const {
        return _mm_or_ps(_mm_dp_ps(row0, vec.data.raw, 0xF1),
               _mm_or_ps(_mm_dp_ps(row1, vec.data.raw, 0xF2),
               _mm_or_ps(_mm_dp_ps(row2, vec.data.raw, 0xF4),
                         _mm_dp_ps(row3, vec.data.raw, 0xF8))));
    }
};

float dot(const vec4f& a, const vec4f& b) {
    __m128 dp = _mm_dp_ps(a.data.raw, b.data.raw, 0xF1);
    return _mm_cvtss_f32(dp);
}

vec4f dot4(const std::array<vec4f,4>& a, const std::array<vec4f, 4>& b) {
    __m128 dp = _mm_or_ps(_mm_dp_ps(a[0].data.raw, b[0].data.raw, 0xF1),
                _mm_or_ps(_mm_dp_ps(a[1].data.raw, b[1].data.raw, 0xF2),
                _mm_or_ps(_mm_dp_ps(a[2].data.raw, b[2].data.raw, 0xF4),
                          _mm_dp_ps(a[3].data.raw, b[3].data.raw, 0xF8))));
    return dp;
}

void testDot() {
    std::cout << "dot: " << dot(vec4f{1.0f, 1.0f, 0.0f, 0.0f}, vec4f{1.0f, 1.0f, 0.0f, 0.0f}) << std::endl;
    std::cout << "dot: " << dot(vec4f{2.0f, 3.0f, 0.0f, 0.0f}, vec4f{5.0f, 5.0f, 0.0f, 0.0f}) << std::endl;
}

void testMat4f() {
    const mat4f id {{1.0f, 0.0f, 0.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f, 0.0f},
                    {0.0f, 0.0f, 1.0f, 0.0f},
                    {0.0f, 0.0f, 0.0f, 1.0f}};

    const mat4f shuffle {{0.0f, 0.0f, 1.0f, 0.0f},
                         {0.0f, 0.0f, 0.0f, 1.0f},
                         {0.0f, 1.0f, 0.0f, 0.0f},
                         {1.0f, 0.0f, 0.0f, 0.0f}};

    const vec4f vec{1.0f, 2.0f, 3.0f, 4.0f};
    const vec4f unchanged = id * vec;
    const vec4f shuffled = shuffle * vec;

    std::cout << "testMat4f: " << unchanged << std::endl;
    std::cout << "testMat4f: " << shuffled << std::endl;
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
    testMat4f();

    return 0;
}
