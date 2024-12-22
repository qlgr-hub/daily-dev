#pragma once

#include <array>
#include <string>
#include <string_view>


// FixedCapacityStack 定义后，用下面的 using 别名代替:
//    template <size_t N>
//    using FixedCapacityStackOfStrings = FixedCapacityStack<std::string, N>;
//
// template <size_t N>
// class FixedCapacityStackOfStrings {
//     std::array<std::string, N> _storage;
//     size_t _size{ 0 };
//
// public:
//     FixedCapacityStackOfStrings() = default;
//
// public:
//     void push(std::string_view elem) {
//         _storage[_size++] = elem;
//     }
//
//     std::string pop() {
//         return _storage[--_size];
//     }
//
//     bool isEmpty() const {
//         return _size == 0;
//     }
//
//     size_t size() const {
//         return _size;
//     }
// };


template <typename T, size_t N>
class FixedCapacityStack {
    std::array<T, N> _storage;
    size_t _size{ 0 };

public:
    FixedCapacityStack() = default;

public:
    void push(const T& elem) {
        _storage[_size++] = elem;
    }

    T pop() {
        return _storage[--_size];
    }

    bool isEmpty() const {
        return _size == 0;
    }

    size_t size() const {
        return _size;
    }
};


template <typename T>
class ResizingArrayStack {
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

public:
    ResizingArrayStack(size_t initCap = 2) {
        _resize(initCap);
    }

    ~ResizingArrayStack() {
        if (_storage) {
            delete [] _storage;
            _storage = nullptr;
        }
    }

public:
    void push(const T& val) {
        if (_N == _cap) {
            _resize(2 * _cap);
        }
        _storage[_N++] = val;
    }

    T pop() {
        auto res = _storage[--_N];

        if (_N > 0 && _N == _cap / 4) {
            _resize(_cap / 2);
        }
        return res;
    }

    size_t size() const {
        return _N;
    }

    bool isEmpty() const {
        return (_N == 0);
    }
};


template <typename T>
class LinkedStack {
    struct _Node;

    struct _Node {
        T _item{ T{} };
        _Node* _next{ nullptr };

        _Node() = default;

        _Node(const T& val) : _item{ val }, _next{ nullptr } {
        }

        _Node(const T& val, _Node* next) : _item{ val }, _next{ next } {
        }

        static T free(_Node** pptr) {
            auto res{ (*pptr)->_item };
            delete (*pptr);
            (*pptr) = nullptr;
            return res;
        }
    };

    _Node* _head{ nullptr };
    size_t _N{ 0 };

public:
    LinkedStack() = default;
    ~LinkedStack() {
        auto p{ _head };
        while (p) {
            auto tmp{ p };
            p = p->_next;
            _Node::free(&tmp);
        }
    }

    bool isEmpty() const {
        return (_head == nullptr);
    }

    size_t size() const {
        return _N;
    }

    void push(const T& val) {
        _Node* oldHead{ _head };
        _head = new _Node{ val, oldHead };
        ++_N;
    }

    T pop() {
        if (_head) {
            auto oldHead{ _head };
            _head = _head->_next;
            --_N;

            return _Node::free(&oldHead);
        }
        return T{};
    }
};
