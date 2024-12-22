#pragma once

#include <cstddef>
#include <vector>

// 其实 C++ 中这个没必要，直接用 std::vector 大不了删除的接口不用了呗
template <typename T>
class Bag {
    std::vector<T> _storage;

public:
    Bag() = default;

    Bag(size_t size, const T& defaultVal = T{}) : _storage(size, defaultVal) {
    }
     
    Bag(std::initializer_list<T> lst) : _storage{ lst } {
    }

public:
    void add(const T& item) {
        _storage.push_back(item);
    }

    auto begin() {
        return _storage.begin();
    }

    auto end() {
        return _storage.end();
    }

    auto cbegin() const {
        return _storage.cbegin();
    }

    auto cend() const {
        return _storage.cend();
    }

    bool empty() const {
        return _storage.empty();
    }

    size_t size() const {
        return _storage.size();
    }

    T& operator[](size_t idx) {
        return _storage[idx];
    }

    const T& operator[](size_t idx) const {
        return _storage[idx];
    }
};

