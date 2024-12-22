#include <iostream>
#include <array>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>

template <typename T, int MAX_SIZE=100>
class Stack {
    std::array<T, MAX_SIZE> S;
    int top;

public:
    Stack() : top{ 0 } {
    }

    bool isEmpty() const {
        return top == 0;
    }

    bool isFull() const {
        return top >= MAX_SIZE;
    }

    int size() const {
        return top + 1;
    }

    void push(T&& x) {
        if (isFull())
            throw std::out_of_range("push to full stack");
        S[++top] = x;
    }

    T pop() {
        if (isEmpty())
            throw std::out_of_range("pop from empty stack");
        return std::move(S[top--]);
    }
};

int main1() {
    Stack<double> ops;

    std::string expr;
    while (std::getline(std::cin, expr)) {
        std::istringstream istr(expr);

        std::string s;
        double a;
        double b;
        while (istr >> s) {
            switch (s[0]) {
            case '+':
                a = ops.pop();
                b = ops.pop();
                ops.push(a + b);
                break;

            case '-':
                a = ops.pop();
                b = ops.pop();
                ops.push(a - b);
                break;

            case '*':
                a = ops.pop();
                b = ops.pop();
                ops.push(a * b);
                break;

            case '/':
                a = ops.pop();
                b = ops.pop();
                ops.push(a / b);
                break;

            default:
                ops.push(std::stod(s));
                break;
            }
        }

        std::cout << ops.pop() << '\n';
        std::cout << ops.size() << '\n';
    }

    return 0;
}

int main2() {
    std::stack<int> S;

    S.push(3);
    S.push(7);
    S.push(1);
    std::cout << S.size() << ' ';

    std::cout << S.top() << ' ';
    S.pop();

    std::cout << S.top() << ' ';
    S.pop();

    std::cout << S.top() << ' ';
    S.push(5);

    std::cout << S.top() << ' ';
    S.pop();

    std::cout << S.top() << '\n';
    return 0;
}

int main() {
    std::stack<double> ops;

    std::string expr;
    while (std::getline(std::cin, expr)) {
        std::istringstream istr(expr);

        std::string s;
        double a;
        double b;
        while (istr >> s) {
            switch (s[0]) {
            case '+':
                a = ops.top(); ops.pop();
                b = ops.top(); ops.pop();
                ops.push(a + b);
                break;

            case '-':
                a = ops.top(); ops.pop();
                b = ops.top(); ops.pop();
                ops.push(a - b);
                break;

            case '*':
                a = ops.top(); ops.pop();
                b = ops.top(); ops.pop();
                ops.push(a * b);
                break;

            case '/':
                a = ops.top(); ops.pop();
                b = ops.top(); ops.pop();
                ops.push(a / b);
                break;

            default:
                ops.push(std::stod(s));
                break;
            }
        }

        std::cout << ops.top() << '\n';
        std::cout << ops.size() << '\n';
    }

    return 0;
}