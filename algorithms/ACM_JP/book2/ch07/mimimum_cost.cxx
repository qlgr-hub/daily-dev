#include <iostream>
#include <algorithm>
#include <array>
#include <vector>

static const int VMAX = 10000;

int solve(const std::vector<int>& A, int s) {
    std::vector<int> B{ A.cbegin(), A.cend() };
    std::vector<bool> V(A.size(), false);

    std::sort(B.begin(), B.end());
    std::vector<int> T(B.size());
    for (int i = 0; i < B.size(); ++i)
        T[B[i]] = i;

    int ans = 0;
    for (int i = 0; i < B.size(); ++i) {
        if (V[i])
            continue;

        int cur = i;
        int S = 0;
        int m = VMAX;
        int an = 0;
        while (true) {
            V[cur] = true;
            an++;
            int v = A[cur];
            m = std::min(m, v);
            S += v;
            cur = T[v];
            if (V[cur])
                break;
        }
        ans += std::min(S + (an - 2) * m, m + S + (an + 1) * s);
    }

    return ans;
}

int main() {
    int n = 0;
    std::cin >> n;

    int s = VMAX;
    std::vector<int> A(n);
    for (int i = 0; i < n; ++i) {
        std::cin >> A[i];
        s = std::min(s, A[i]);
    }

    int ans = solve(A, s);
    std::cout << ans << std::endl;

    return 0;
}
