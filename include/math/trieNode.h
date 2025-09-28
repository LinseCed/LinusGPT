#pragma once
#include <unordered_map>
#include <cstdint>

struct TrieNode {
    bool isToken = false;
    std::unordered_map<uint8_t, TrieNode*> children;

    ~TrieNode() {
        for (auto& kv : children) {
            delete kv.second;
        }
    }
};
