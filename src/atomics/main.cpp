#include <atomic>
#include <stdint.h>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace {
    auto random() {
        static std::random_device device;
        static std::default_random_engine engine(device());
        std::uniform_int_distribution<int> uniform_dist(1, 17);
        return uniform_dist(engine);
    }

    constexpr size_t NUM_ITERATIONS = 10000;
    std::atomic<uint32_t> atmc {0};
    constexpr size_t NUM_THREADS = 16u;
    std::vector<std::vector<uint32_t>> results{NUM_THREADS, std::vector<uint32_t>(NUM_ITERATIONS, 0xffffffffu)};

    void thread_func(size_t idx) {
        auto &res = results[idx];
        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            //if (i % 1000 == 0) std::cout << "Thread " << idx << " iteration" << i << std::endl;
            uint32_t size = random();
            res[i] = atmc.fetch_add(size, std::memory_order_relaxed);
            auto end = std::chrono::steady_clock::now() + std::chrono::nanoseconds(1);
            while (std::chrono::steady_clock::now() < end);
        }
    }
}

int main() {
    std::vector<std::jthread> threads;

    for (size_t i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(thread_func, i);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    uint32_t last = 0xffffffffu;
    std::vector<size_t> indices(NUM_THREADS, 0u);
    for (size_t i = 0; i < NUM_ITERATIONS * NUM_THREADS; ++i) {
        size_t min_idx = 0;
        uint32_t min = 0xffffffffu;
        for (size_t j = 0; j < NUM_THREADS; ++j) {
            if (indices[j] == NUM_ITERATIONS) continue;
            auto& val = results[j][indices[j]];
            if (val < min) {
                min = val;
                min_idx = j;
            }
        }
        ++indices[min_idx];
        //std::cout << min_idx << ":" << min << " ";
        if (min == last) {
            std::cout << "Found duplicate: " << last << "\n";
        }
        last = min;
    }

    std::cout << "\n" << atmc << std::endl;

    return 0;
}
