#include <iostream>
#include <queue>

int main() {
    std::queue<int> q;
    q.push(1);
    q.push(2);
    q.push(3);
    std::cout << q.front() << "\n";
    q.pop();
    std::cout << q.front() << "\n";
    q.pop();
    std::cout << q.front() << "\n";
    q.pop();
    return 0;
}