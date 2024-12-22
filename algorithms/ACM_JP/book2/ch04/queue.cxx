#include <cstdlib>
#include <iostream>
#include <array>
#include <queue>
#include <string>
#include <utility>

template <typename T, int MAX_SIZE = 100>
class Queue {
    std::array<T, MAX_SIZE> Q;
    int head;
    int tail;

public:
    Queue() : head{ 0 }, tail{ 0 } {
    }

    void enqueue(T&& x) {
        Q[tail] = x;
        tail = (tail + 1) % MAX_SIZE;
    }

    T dequeue() {
        T x = Q[head];
        head = (head + 1) % MAX_SIZE;
        return std::move(x);
    }

    bool isEmpty() const {
        return head == tail;
    }
};

struct P {
    std::string name;
    int time;
};

int main1() {
    int N = 0, q = 0;
    std::cin >> N >> q;

    Queue<P> taskQ;
    for (int i = 0; i < N; ++i) {
        P x;
        std::cin >> x.name >> x.time;
        taskQ.enqueue(std::move(x));
    }
    std::cout << '\n';

    int elaps = 0;
    while (!taskQ.isEmpty()) {
        P u = taskQ.dequeue();
        int c = std::min(u.time, q);
        u.time -= c;
        elaps += c;

        if (u.time > 0) {
            taskQ.enqueue(std::move(u));
        }
        else {
            std::cout << u.name << ' ' << elaps << '\n';
        }
    }

    return 0;
}

int main2() {
    std::queue<std::string> Q;

    Q.push("red");
    Q.push("yellow");
    Q.push("yellow");
    Q.push("blue");

    std::cout << Q.front() << ' ';
    Q.pop();

    std::cout << Q.front() << ' ';
    Q.pop();

    std::cout << Q.front() << ' ';
    Q.pop();

    Q.push("green");

    std::cout << Q.front() << ' ';
    Q.pop();

    std::cout << Q.front() << '\n';
    return 0;
}

int main() {
    int N = 0, q = 0;
    std::cin >> N >> q;

    std::queue<std::pair<std::string, int>> taskQ;
    for (int i = 0; i < N; ++i) {
        std::string name;
        int time;
        std::cin >> name >> time;
        taskQ.push(std::make_pair(name, time));
    }
    std::cout << '\n';

    int elaps = 0;
    while (!taskQ.empty()) {
        auto u = std::move(taskQ.front());
        taskQ.pop();
        int a = std::min(u.second, q);
        u.second -= a;
        elaps += a;

        if (u.second > 0) {
            taskQ.push(u);
        }
        else {
            std::cout << u.first << ' ' << elaps << '\n';
        }
    }

    return 0;
}
