#include <iostream>

template <typename T>
void typesafe_print_variadic(T next) {
    std::cout << next << std::endl;
}

template <typename T, typename... Args>
void typesafe_print_variadic(T next, Args... args) {
    std::cout << next << std::endl;
    typesafe_print_variadic(args...);
}

template <typename... Ts>
struct tuple {};

template <typename T, typename... Ts>
struct tuple<T, Ts...> : public tuple<Ts...> {
    tuple(T t, Ts... ts) : tuple<Ts...>{ts...}, value{t} {};
    T value;
};

template <size_t n, typename>
struct element_holder {};

template <typename T, typename... Ts>
struct element_holder<0, tuple<T, Ts...>> {
    using type = T;
};

template <size_t n, typename T, typename... Ts>
struct element_holder<n, tuple<T, Ts...>> {
    using type = typename element_holder<n-1, tuple<Ts...>>::type;
};

template <size_t n, typename... Ts>
std::enable_if_t<n == 0, typename element_holder<0, tuple<Ts...>>::type&> get(tuple<Ts...>& tup) {
    return tup.value;
}

template <size_t n, typename T, typename... Ts>
std::enable_if_t<n != 0, typename element_holder<n, tuple<T, Ts...>>::type&> get(tuple<T, Ts...>& tup) {
    tuple<Ts...>& base = tup;
    return get<n-1>(base);
}

int main() {
    typesafe_print_variadic("1", 1.23f, 1337u, 'a');
    tuple<int, float, char> tup{1337, 0.42f, 'f'};
    std::cout << get<2>(tup) << std::endl;
    return 0;
}
