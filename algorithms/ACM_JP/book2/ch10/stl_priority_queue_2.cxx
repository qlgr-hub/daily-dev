#include <iostream>
#include <string>
#include <queue>

int main() {
    std::priority_queue<int> q;

    while (true) {
        std::string com;
        std::cin >> com;
        if (com == "end")
            break;
        else if (com == "insert") {
            int key;
            std::cin >> key;
            q.push(key);
        }
        else {
            std::cout << q.top() << '\n';
            q.pop();
        }
    }

    return 0;
}