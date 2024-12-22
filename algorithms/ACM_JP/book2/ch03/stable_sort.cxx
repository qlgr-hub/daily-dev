#include <algorithm>
#include <iostream>
#include <vector>

struct Card {
    char suit;
    char value;
};

bool operator==(const Card& lhs, const Card& rhs) {
    return (lhs.suit == rhs.suit) && (lhs.value == rhs.value);
}

bool operator!=(const Card& lhs, const Card& rhs) {
    return !(lhs == rhs);
}

bool operator<(const Card& lhs, const Card& rhs) {
    return lhs.value < rhs.value;
}

bool operator>=(const Card& lhs, const Card& rhs) {
    return !(lhs < rhs);
}

bool operator>(const Card& lhs, const Card& rhs) {
    return lhs.value > rhs.value;
}

bool operator<=(const Card& lhs, const Card& rhs) {
    return !(lhs > rhs);
}

template <typename T>
void bubble(std::vector<T>& A) {
    for (auto it_o = A.begin(); it_o != A.end(); ++it_o) {
        for (auto it_i = it_o + 1; it_i != A.end(); ++it_i) {
            if (*it_i < *it_o) {
                std::iter_swap(it_i, it_o);
            }
        }
    }
}

template <typename T>
void selection(std::vector<T>& A) {
    for (auto it = A.begin(); it != A.end(); ++it) {
        std::iter_swap(it, std::min_element(it, A.end()));
    }
}

int main() {
    int N = 0;
    std::cin >> N;

    std::vector<Card> cards(N);
    for (auto& c : cards) {
        std::cin >> c.suit >> c.value;
    }

    std::vector<Card> cards2{ cards };

    bubble(cards);
    selection(cards2);

    auto print = [](const std::vector<Card>& C) { 
        for (const auto& c : C) {
            std::cout << c.suit << c.value << ' ';
        }
        std::cout << '\n';
    };

    auto isEqual = [](const std::vector<Card>& A, const std::vector<Card>& B) {
        if (A.size() != B.size())
            return false;

        bool e = true;
        for (int i = 0; i < A.size(); ++i) {
            if (A[i] != B[i]) {
                e = false;
                break;
            }
        }
        return e;
    };

    print(cards);
    std::cout << "Stable" << '\n';

    print(cards2);
    if (isEqual(cards, cards2)) {
        std::cout << "Stable" << '\n';
    }
    else {
        std::cout << "Not stable" << '\n';
    }

    return 0;
}