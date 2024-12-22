#ifndef _BINARY_SEARCH_TREE_H_
#define _BINARY_SEARCH_TREE_H_

#include <iostream>

template <typename T>
class BSTree {
    struct Node;

    struct Node {
        T key;
        Node* parent;
        Node* left;
        Node* right;

        Node() : key{ 0 }, parent{ nullptr }, left{ nullptr }, right{ nullptr } {
        }

        Node(const T& k) : key{ k }, parent{ nullptr }, left{ nullptr }, right{ nullptr } {
        }
    };

    Node* root;

    void _preorder(Node* r) const {
        if (r == nullptr)
            return;
        std::cout << ' ' << r->key;
        _preorder(r->left);
        _preorder(r->right);
    }

    void _inorder(Node* r) const {
        if (r == nullptr)
            return;
        _inorder(r->left);
        std::cout << ' ' << r->key;
        _inorder(r->right);
    }

    void _postorder(Node* r) const {
        if (r == nullptr)
            return;
        _postorder(r->left);
        _postorder(r->right);
        std::cout << ' ' << r->key;
    }

    Node* _find(Node* r, int k) {
        while (r != nullptr && k != r->key) {
            if (k < r->key)
                r = r->left;
            else
                r = r->right;
        }
        return r;
    }

    Node* minimum(Node* x) {
        while (x->left != nullptr)
            x = x->left;
        return x;
    }

    Node* _successor(Node* x) {
        if (x->right != nullptr)
            return minimum(x->right);

        Node* y = x->parent;
        while (y != nullptr && x == y->right) {
            x = y;
            y = y->parent;
        }

        return y;
    }

    void _delete(Node* z) {
        Node* y = nullptr;
        if (z->left == nullptr || z->right == nullptr)
            y = z;
        else
            y = _successor(z);

        Node* x = nullptr;
        if (y->left != nullptr)
            x = y->left;
        else
            x = y->right;

        if (x != nullptr)
            x->parent = y->parent;

        if (y->parent == nullptr)
            root = x;
        else {
            if (y == y->parent->left) {
                y->parent->left = x;
            }
            else {
                y->parent->right = x;
            }
        }

        if (y != z) {
            z->key = y->key;
        }

        delete y;
        y = nullptr;
        x = nullptr;
    }

public:
    BSTree() : root{ nullptr } {
    }

    void insert(const T& k) {
        Node* z = new Node{ k };

        Node* y = nullptr;
        Node* x = root;
        while (x != nullptr) {
            y = x;
            if (z->key < x->key)
                x = x->left;
            else
                x = x->right; 
        }

        z->parent = y;
        if (y == nullptr) 
            root = z;
        else {
            if (z->key < y->key) {
                z->left = y->left;
                y->left = z;
            }
            else {
                z->right = y->right;
                y->right = z;
            }
        }
    }

    void preorder() const {
        _preorder(root);
        std::cout << '\n';
    }

    void inorder() const {
        _inorder(root);
        std::cout << '\n';
    }

    void postorder() const {
        _postorder(root);
        std::cout << '\n';
    }

    bool find(int k) {
        Node* u = _find(root, k);
        return (u != nullptr);
    }

    void del(int k) {
        Node* u = _find(root, k);
        if (u) {
            _delete(u);
        }
    }
};

#endif /* !_BINARY_SEARCH_TREE_H_ */
