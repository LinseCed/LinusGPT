#pragma once
#include <unordered_map>
#include <cstdint>
#include <ranges>

struct TrieNode {
    bool isToken = false;
    std::unordered_map<uint8_t, TrieNode*> children;

    ~TrieNode() {
        for (const auto &val: children | std::views::values) {
            delete val;
        }
    }
};
