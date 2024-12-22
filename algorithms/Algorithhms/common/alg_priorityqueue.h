#pragma once

#include <cstddef>
#include <stdexcept>
#include <algorithm>


template <typename T, typename Cmp, bool VER1 = true>
class ResizingArrayPQ {
    T* _storage{ nullptr };
    size_t _cap{ 0 };
    size_t _N{ 0 };

private:
    void _resize(size_t n) {
        _cap = n;
        T* temp = new T[n];
        for (int i{ 0 }; i < _N; ++i) {
            temp[i] = _storage[i];
        }
        delete [] _storage;
        _storage = temp;
    }

    void _insertVer1(const T& key) {
        if (_N == _cap)
            _resize((_cap == 0) ? 2 : _cap * 2);
        _storage[_N++] = key;
    }

    T _topVer1() const {
        if (_N == 0)
            throw std::runtime_error{ "empty PQ" };

        // 选择最值放到最后面
        auto topIdx{ 0 };
        for (int i{ 1 }; i < _N; ++i) {
            if (Cmp{}(_storage[topIdx], _storage[i]))
                topIdx = i;
        }
        std::swap(_storage[_N - 1], _storage[topIdx]);
        return _storage[_N - 1];
    }

    void _insertVer2(const T& key) {
        if (_N == _cap)
            _resize((_cap == 0) ? 2 : _cap * 2);

        // 插入到有序序列中
        auto j{ static_cast<int>(_N) - 1 };
        while (j >= 0 && Cmp{}(key, _storage[j])) {
            _storage[j + 1] = _storage[j];
            --j;
        }
        _storage[j + 1] = key;
        ++_N;
    }

    T _topVer2() const {
        if (_N == 0)
            throw std::runtime_error{ "empty PQ" };
        return _storage[_N - 1];
    }

public:
    ResizingArrayPQ() = default;

    ResizingArrayPQ(int max) {
        max = (max <= 0) ? 2 : max;
        _resize(max);
    }

public:
    void insert(const T& key) {
        if constexpr (VER1) {
            _insertVer1(key);
        }
        else {
            _insertVer2(key);
        }
    }

    T top() const {
        if constexpr (VER1) {
            return _topVer1();
        }
        else {
            return _topVer2();
        }
    }

    T delTop() {
        auto val{ top() };
        if (--_N == _cap / 4)
            _resize(_cap / 2);
        return val;
    }

    bool empty() const {
        return (_N == 0);
    }

    std::size_t size() const {
        return _N;
    }
};


template <typename T, typename Cmp>
class HeapPQ {
    std::vector<T> _storage;
    size_t _N;

private:
    void _swim(int k) {
        while (k > 1 && Cmp{}(_storage[k / 2], _storage[k])) {
            std::swap(_storage[k / 2], _storage[k]);
            k = k / 2;
        }
    }

    void _sink(int k) {
        while (2 * k <= _N) {
            int j{ 2 * k };
            if (j < _N && Cmp{}(_storage[j], _storage[j + 1]))
                ++j;
            if (!Cmp{}(_storage[k], _storage[j]))
                break;
            std::swap(_storage[k], _storage[j]);
            k = j;
        }
    }

public:
    HeapPQ(int max) : _storage(max + 1, T()), _N{ 0 } {
    }

public:
    void insert(const T& key) {
        if (_N == _storage.size() - 1)
            throw std::runtime_error{ "full PQ" };

        _storage[++_N] = key;
        _swim(_N);
    }

    T top() const {
        if (_N == 0)
            throw std::runtime_error{ "empty PQ" };
        return _storage[1];
    }

    T delTop() {
        auto curTop{ top() };
        std::swap(_storage[1], _storage[_N--]);
        _storage[_N + 1] = T();
        _sink(1);
        return curTop;
    }

    bool empty() const {
        return (_N == 0);
    }

    std::size_t size() const {
        return _N;
    }
};


// 参考自：https://github.com/jimmysuncpt/Algorithms/tree/master/src/edu/princeton/cs/algs4
template <typename T, typename Cmp>
class IndexPQ {
    std::vector<T> _keys;
    std::vector<int> _pq;
    std::vector<int> _qp;
    size_t _N;

private:
    bool _cmp(int i, int j) {
        return Cmp{}(_keys[_pq[i]], _keys[_pq[j]]);
    }

    void _exch(int i, int j) {
        auto temp{ _pq[i] };
        _pq[i] = _pq[j];
        _pq[j] = temp;
        _qp[_pq[i]] = i;
        _qp[_pq[j]] = j;
    }

    void _swim(int k) {
        while (k > 1 && _cmp(k / 2, k)) {
            _exch(k / 2, k);
            k = k / 2;
        }
    }

    void _sink(int k) {
        while (2 * k <= _N) {
            int j{ 2 * k };
            if (j < _N && _cmp(j, j + 1))
                ++j;
            if (!_cmp(k, j))
                break;
            _exch(k, j);
            k = j;
        }
    }

public:
    IndexPQ(int max) : _keys(max + 1, T()), _pq(max + 1, -1), _qp(max + 1, -1), _N{ 0 } {
    }

public:
    bool contains(int i) {
        if (i < 0 || i >= _keys.size())
            throw std::invalid_argument{ "index out of range" };
        return (_qp[i] != -1);
    }

    void change(int i, const T& key) {
        if (i < 0 || i >= _keys.size())
            throw std::invalid_argument{ "index out of range" };
        if (!contains(i))
            throw std::invalid_argument{ "no such element" };
        _keys[i] = key;
        _swim(_pq[i]);
        _sink(_pq[i]);
    }

    void insert(int i, const T& key) {
        if (i < 0 || i >= _keys.size())
            throw std::invalid_argument{ "index out of range" };
        if (contains(i))
            throw std::invalid_argument{ "index already exists" };
        ++_N;
        _qp[i] = _N;
        _pq[_N] = i;
        _keys[i] = key;
        _swim(_N);
    }

    T top() const {
        if (_N == 0)
            throw std::runtime_error{ "empty PQ" };
        return _keys[_pq[1]];
    }

    int delTop() {
        auto topIdx{ _pq[1] };
        _exch(1, _N--);
        _sink(1);
        _qp[topIdx] = -1;
        _keys[_pq[_N + 1]] = T();
        _pq[_N + 1] = -1;
        return topIdx;
    }

    bool empty() const {
        return (_N == 0);
    }

    std::size_t size() const {
        return _N;
    }
};
