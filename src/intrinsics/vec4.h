#pragma once
#include <array>
#include <iostream>
#include <string>
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
