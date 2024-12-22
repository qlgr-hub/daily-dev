#include <fstream>
#include <iostream>
#include <vector>

struct Card {
    char suit;
    int value;
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

std::ofstream ofs("./1.txt");

int partition(std::vector<Card>& A, int p, int r) {
    Card x = A[r];
    int i = p - 1;
    for (int j = p; j < r; ++j) {
        if (A[j] <= x) {
            std::swap(A[j], A[++i]);
        }
    }
    std::swap(A[r], A[++i]);
    return i;
}

void quickSort(std::vector<Card>& A, int p, int r) {
    if (p < r) {
        int q = partition(A, p, r);
        ofs << p << " " << r << " " << q << '\n';
        quickSort(A, p, q - 1);
        quickSort(A, q + 1, r);
    }
}

template <typename T>
void bubbleSort(std::vector<T>& A) {
    for (auto it_o = A.begin(); it_o != A.end(); ++it_o) {
        for (auto it_i = it_o + 1; it_i != A.end(); ++it_i) {
            if (*it_i < *it_o) {
                std::iter_swap(it_i, it_o);
            }
        }
    }
}

int main() {
    int n = 0;
    std::cin >> n;

    std::vector<Card> A(n);
    for (int i = 0; i < n; ++i)
        std::cin >> A[i].suit >> A[i].value;

    std::vector<Card> B{A.cbegin(), A.cend()};

    quickSort(A, 0, n - 1);
    bubbleSort(B);

    bool stable = true;
    for (int i = 0; i < n; ++i) {
        if (A[i].suit != B[i].suit) {
            stable = false;
            break;
        }
    }
    
    std::cout << (stable ? "Stable\n" : "Not stable\n");

    for (int i = 0; i < n; ++i)
        std::cout << A[i].suit << ' ' << A[i].value << '\n';

    return 0;
}
