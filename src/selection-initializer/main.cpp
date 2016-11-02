#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>

auto create_uniform_distribution(const int lower, const int upper) {
    std::random_device device;
    std::default_random_engine engine(device());

    std::uniform_int_distribution<int> uniform_dist(lower, upper);
    return std::bind(uniform_dist, engine);
}

int main() {
    std::unordered_map<size_t, std::string> intToString{{0, "zero"}, {1, "one"}, {42, "forty two"}};

    if (auto it = intToString.find(42); it != end(intToString)) {
        std::cout << it->first << ": " << it->second << std::endl;
    }

    if (auto it = intToString.find(1337); it != end(intToString)) {
        std::cout << it->first << ": " << it->second << std::endl;
    }

    switch (auto dist = create_uniform_distribution(0, 3); dist()) {
        case 0:
            std::cout << "case 0" << std::endl;
            break;
        case 1:
            std::cout << "case 1" << std::endl;
            break;
        case 2:
            std::cout << "case 2" << std::endl;
            break;
        case 3:
            std::cout << "case 3" << std::endl;
            break;
    }
}
