#include <iostream>

template <typename T>
void typesafe_print_variadic(const char * fmt, T next) {
    std::cout << next << std::endl;
}

template <typename T, typename... Args>
void typesafe_print_variadic(const char * fmt, T next, Args... args) {
    std::cout << next << std::endl;
    typesafe_print_variadic(fmt, args...);
}

template <typename... Ts>
struct tuple {};

template <typename T, typename... Ts>
struct tuple<T, Ts...> : public tuple<Ts...> {
    tuple(T t, Ts... ts) : tuple<Ts...>{ts...}, value{t} {};
    T value;
};

template <size_t k, typename>
struct element_holder {};

template <typename T, typename... Ts>
struct element_holder<0, tuple<T, Ts...>> {
    using type = T;
};

template <size_t n, typename T, typename... Ts>
struct element_holder<n, tuple<T, Ts...>> {
    using type = typename element_holder<n-1, tuple<Ts...>>::type;
};

int main() {
    typesafe_print_variadic("1", 1.23f, 1337u, 'a');
    tuple<int, float, char> tup{1337, 0.42f, 'f'};
    std::cout << tup.value << std::endl;
    return 0;
}
