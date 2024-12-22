#include <fstream>
#include <queue>
#include <stack>
#include <iostream>
#include <cmath>
#include "../common/alg_bag.h"
#include "../common/alg_stack.h"
#include "../common/alg_queue.h"


class Stats {
    static Bag<int> _readInts(std::string_view filename) {
        std::ifstream in(std::move(std::string{ filename }));

        // 只是个例子 
        std::queue<int> q;
        std::string str;
        while (in >> str) {
            if (str.empty())
                continue;

            q.push(std::stoi(str));
        }

        Bag<int> res(q.size());
        for (auto& elem : res) {
            elem = q.front();
            q.pop();
        }

        return res;
    }

    static Bag<int> _readInts() {
        Bag<int> res{};

        int num{ 0 };
        while (std::cin >> num) {
            res.add(num);
        }

        return res;
    }

public:
    static void test(int argc, char* argv[]) {
        Bag<int> numbers { ((argc > 1) ? _readInts(argv[1]) : _readInts()) };
        size_t N{ numbers.size() };

        double sum{ 0.0 };
        for (const auto& x : numbers) {
            sum += x;
        }
        double mean{ sum / N };

        sum = 0.0;
        for (const auto& x : numbers) {
            sum += (x - mean) * (x - mean);
        }
        double stdDev = std::sqrt(sum / (N - 1));

        std::cout.flags(std::ios::fixed);
        std::cout.precision(2);
        std::cout << "Mean: " << mean << '\n';
        std::cout << "Std dev: " << stdDev << '\n';
    }
};


class Reverse {
public:
    static void test(int argc, char* argv[]) {
        std::stack<int> s;

        int num{ 0 };
        while (std::cin >> num) {
            s.push(num);
        }

        while (!s.empty()) {
            std::cout << s.top() << ' ';
            s.pop();
        }
        std::cout << '\n';
    }
};


class Evaluate {
public:
    static void test(int argc, char* argv[]) {
        std::stack<std::string> ops;
        std::stack<double> vals;

        std::string str;
        while (std::cin >> str) {
            if (str == "(") {
                // do nothing
            }
            else if (str == "+" || str == "-" || str == "*" || str == "/" || str == "sqrt") {
                ops.push(str);
            }
            else if (str == ")") {
                std::string op{ ops.top() }; ops.pop();
                double val{ vals.top() }; vals.pop();
                if (op == "+") {
                    val = vals.top() + val; vals.pop();
                }
                else if (op == "-") {
                    val = vals.top() - val; vals.pop();
                }
                else if (op == "*") {
                    val = vals.top() * val; vals.pop();
                }
                else if (op == "/") {
                    val = vals.top() / val; vals.pop();
                }
                else if (op == "sqrt") {
                    val = std::sqrt(val);
                }
                vals.push(val);
            }
            else {
                vals.push(std::stod(str));
            }
        }

        std::cout.flags(std::ios::fixed);
        std::cout.precision(15);
        std::cout << vals.top() << '\n';
    }
};

template <size_t N>
using FixedCapacityStackOfStrings = FixedCapacityStack<std::string, N>;

class StackDemo {
public:
    static void test(int argc, char* argv[]) {
        // FixedCapacityStackOfStrings<100> s;       // 固定大小字符串栈
        // FixedCapacityStack<std::string, 100> s;   // 固定大小栈（本测试存字符串）
        // ResizingArrayStack<std::string> s;        // 可调整大小的栈（本测试存字符串）
        LinkedStack<std::string> s;                  // 链式栈（本测试存字符串）

        std::string str;
        while (std::cin >> str) {
            if (str != "-") {
                s.push(str);
            }
            else if (!s.isEmpty()) {
                std::cout << s.pop() << ' ';
            }
        }

        std::cout << "(" << s.size() << " left on stack)\n";
    }
};


class QueueDemo {
public:
    static void test(int argc, char* argv[]) {
        LinkedQueue<std::string> s;  // 链式队列（本测试存字符串）
        std::string str;
        while (std::cin >> str) {
            if (str != "-") {
                s.enqueue(str);
            }
            else if (!s.isEmpty()) {
                std::cout << s.dequeue() << ' ';
            }
        }

        std::cout << "(" << s.size() << " left on queue)\n";
    }
};


int main(int argc, char* argv[]) {
    // Stats::test(argc, argv);

    // Reverse::test(argc, argv);

    // Evaluate::test(argc, argv);

    // StackDemo::test(argc, argv);

    QueueDemo::test(argc, argv);

    return 0;
}