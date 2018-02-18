#include <iostream>
#include <string>
#include <vector>

using std::vector;
using std::string;
using std::cout;
using std::cin;
using std::endl;

template <typename T>
std::ostream& operator<<(std::ostream& os, const vector<T>& vec) {
    string sep = "";
    os << "{";
    for (const auto& el : vec) {
        os << sep << el;
        sep = ", ";
    }
    os << "}";
    return os;
}

template <typename T>
void test_operators(const vector<T>& lhs, const vector<T>& rhs) {
    cout << "vectors: " << lhs << ", " << rhs << endl;
    cout << "lhs == rhs " << (lhs == rhs ? "true" : "false") << endl;
    cout << "lhs != rhs " << (lhs != rhs ? "true" : "false") << endl;
    cout << "lhs < rhs " << (lhs < rhs ? "true" : "false") << endl;
    cout << "lhs <= rhs " << (lhs <= rhs ? "true" : "false") << endl;
    cout << "lhs > rhs " << (lhs > rhs ? "true" : "false") << endl;
    cout << "lhs >= rhs " << (lhs >= rhs ? "true" : "false") << endl;
}

int main() {
    test_operators(vector<string>{"a", "b", "c"},
                   vector<string>{"a", "b", "b"});
    test_operators(vector<string>{"a", "b", "c"},
                   vector<string>{"aa", "b", "b"});
    test_operators(vector<string>{"a", "b", "c"},
                   vector<string>{"a", "bb", "b"});
    test_operators(vector<string>{"a", "bb", "c"},
                   vector<string>{"aa", "b", "b"});
    test_operators(vector<string>{"a", "b", "c"},
                   vector<string>{"b", "a", "b"});
    return 0;
}
