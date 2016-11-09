#include <iostream>
#include "immintrin.h"

union vec4 {
    __m256d raw;
    double val[4];
};

int main() {
    __m256d v1 = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
    __m256d v2 = _mm256_set_pd(2.0, 3.0, 4.0, 5.0);

    vec4 res = {_mm256_mul_pd(v1, v2)};

    for (double val : res.val) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

