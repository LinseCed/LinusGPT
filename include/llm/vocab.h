#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "math/trieNode.h"

class Vocab {
public:
    static Vocab& getInstance();

    [[nodiscard]] int getVocabSize() const;
    [[nodiscard]] const std::vector<std::string>& getVocab() const;

    [[nodiscard]] std::vector<std::string> tokenize(const std::string& text) const;
    [[nodiscard]] std::vector<int> encode(const std::string& text) const;
private:
    Vocab();
    void readVocabFromFile(const std::string& filename);

    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> tokenToId;
    TrieNode* trieRoot;
    
    static const std::vector<uint8_t> MARKER;
};
