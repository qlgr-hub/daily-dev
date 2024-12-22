#include <iostream>
#include <queue>

int main() {
    std::priority_queue<int> PQ;

    PQ.push(1);
    PQ.push(8);
    PQ.push(3);
    PQ.push(5);

    std::cout << PQ.top() << ' ';
    PQ.pop();

    std::cout << PQ.top() << ' ';
    PQ.pop();

    PQ.push(11);

    std::cout << PQ.top() << ' ';
    PQ.pop();

    std::cout << PQ.top() << '\n';
    PQ.pop();

    return 0;
}