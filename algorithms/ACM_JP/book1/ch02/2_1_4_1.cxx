#include <iostream>
#include <vector>

bool dfs(const std::vector<int>& a, int k, int i, int sum) {
    if (i == a.size())
        return (sum == k);

    if (dfs(a, k, i + 1, sum)) 
        return true;

    if (dfs(a, k, i + 1, sum + a[i]))
        return true;

    return false;
}

int main() {
    int n, k;
    std::cin >> n >> k;

    std::vector<int> arr(n);
    for (int i = 0; i < arr.size(); ++i) {
        std::cin >> arr[i];
    }

    std::cout << (dfs(arr, k, 0, 0) ? "Yes" : "No") << std::endl;
    return 0;
}