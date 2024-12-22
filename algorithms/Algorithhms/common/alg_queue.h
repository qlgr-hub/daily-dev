#pragma once

#include <cstddef>

template <typename T>
class LinkedQueue {
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
    _Node* _last{ nullptr };
    size_t _N{ 0 };

public:
    LinkedQueue() = default;
    ~LinkedQueue() {
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

    void enqueue(const T& val) {
        _Node* oldLast{ _last };
        _last = new _Node{ val };
        if (isEmpty()) {
            _head = _last;
        }
        else {
            oldLast->_next = _last;
        }
        ++_N;
    }

    T dequeue() {
        if (_head) {
            auto oldHead{ _head };
            _head = _head->_next;
            --_N;
            if (isEmpty()) {
                _last = nullptr;
            }

            return _Node::free(&oldHead);
        }
        return T{};
    }
};
