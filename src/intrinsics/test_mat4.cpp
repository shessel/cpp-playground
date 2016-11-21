#include <iostream>
#include "immintrin.h"
#include "smmintrin.h"
#include "mat4.h"

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

int main() {
    testMat4f();
    return 0;
}
