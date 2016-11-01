#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7};
    const int factor = 5;
    std::for_each(std::begin(vec), std::end(vec), [=](int &x){ x *= factor; });
    std::for_each(std::begin(vec), std::end(vec), [](int x){ std::cout << x << " "; });
    std::cout << std::endl;
}
