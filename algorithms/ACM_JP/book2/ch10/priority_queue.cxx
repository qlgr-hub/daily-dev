#include <iostream>
#include <string>
#include "pq.h"

int main() {
    PQ<int> q;

    while (true) {
        std::string com;
        std::cin >> com;
        if (com == "end")
            break;
        else if (com == "insert") {
            int key;
            std::cin >> key;
            q.insert(key);
        }
        else {
            std::cout << q.extract() << '\n';
        }
    }

    return 0;
}