#include <chrono>
#include <numeric>
#include <iostream>
#include <type_traits>
#include <vector>

using clock_type = typename std::conditional< std::chrono::high_resolution_clock::is_steady,
											  std::chrono::high_resolution_clock,
											  std::chrono::steady_clock >::type ;

template <typename T>
void aliased(T *a, T *b, T *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = b[i] + c[i];
        b[i] = b[i] + c[i];
    }
}

template <typename T>
void nonAliased(T * __restrict__ a, T * __restrict__ b, T * __restrict__ c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = b[i] + c[i];
        b[i] = b[i] + c[i];
    }
}

template <typename... Args>
void measure(size_t numReps, void (*func)(Args...), Args... args) {
    func(args...);

    std::clock_t start = std::clock();
	const auto n_start = clock_type::now();

    for (size_t i = 0; i < numReps; i++) {
      func(args...);
    }

    std::clock_t end = std::clock();
	const auto n_end = clock_type::now();
    std::cout << start << " " << end << std::endl;
    //std::cout << n_start.count() << " " << n_end.count() << std::endl;
    std::cout << CLOCKS_PER_SEC << std::endl;
    std::cout << static_cast<float>(end - start) / CLOCKS_PER_SEC << std::endl;
	std::cout << std::chrono::duration_cast<std::chrono::seconds>(n_end - n_start).count() << std::endl;
} 

void test() {
    std::cout << "Hello" << std::endl;
}

int main() {
    using DataType = int;
    static const size_t N = 1 << 20;

    std::vector<DataType> a(N);
    std::iota(std::begin(a), std::end(a), 0);

    std::vector<DataType> b(N);
    std::iota(std::rbegin(b), std::rend(b), 0);

    std::vector<DataType> c(N);
    std::iota(std::begin(c), std::end(c), 0);

    measure(1, aliased, a.data(), b.data(), c.data(), N);
    nonAliased(a.data(), b.data(), c.data(), N);

    measure(100, test);

    return 0;
}
