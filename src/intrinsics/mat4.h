#pragma once
#include "immintrin.h"
#include "smmintrin.h"
#include "vec4.h"

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
