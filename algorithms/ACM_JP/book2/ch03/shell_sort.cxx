#include <iostream>
#include <vector>

int insertion_sort(std::vector<int>& A, int g) {
    int cnt = 0;

    for (int i = g; i < A.size(); ++i) {
        int v = A[i];
        int j = i - g;

        while (j >= 0 && A[j] > v) {
            A[j + g] = A[j];
            j -= g;
            ++cnt;
        }

        A[j + g] = v;
    }

    return cnt;
}

void shell_sort(std::vector<int>& A) {
    std::vector<int> G;
    for (int h = 1; h < A.size(); h = 3*h + 1) {
        G.emplace_back(h);
    }

    int cnt = 0;
    for (int i = G.size() - 1; i >= 0; --i) {
        cnt += insertion_sort(A, G[i]);
    }

    // diagnostics
    std::cout << G.size() << '\n';
    for (auto g : G)
        std::cout << g << ' ';
    std::cout << '\n';
    std::cout << cnt << '\n';
}

int main() {
    int N = 0;
    std::cin >> N;

    std::vector<int> A(N);
    for (auto& v : A) {
        std::cin >> v;
    }

    shell_sort(A);

    auto print = [](const std::vector<int>& V) {
        for (auto v : V) {
            std::cout << v << '\n';
        }
    };
    print(A);

    return 0;
}
