#include <iostream>
#include <stack>

int main() {
    std::stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    std::cout << s.top() << "\n";
    s.pop();
    std::cout << s.top() << "\n";
    s.pop();
    std::cout << s.top() << "\n";
    s.pop();
    return 0;
}