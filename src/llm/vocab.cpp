#include "llm/vocab.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>

const std::vector<uint8_t> Vocab::MARKER = {0xE2, 0x96, 0x81};

Vocab& Vocab::getInstance() {
    static Vocab instance;
    return instance;
}

Vocab::Vocab() {
    trieRoot = new TrieNode();
    readVocabFromFile("resources/llmmodel.vocab");
}

void Vocab::readVocabFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open vocab file " + filename);
    }
    std::string line;
    int id = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        vocab.push_back(token);
        tokenToId[token] = id++;

        TrieNode* node = trieRoot;
        std::vector<uint8_t> tokenBytes(token.begin(), token.end());
        for (uint8_t byte : tokenBytes) {
            if (!node->children.contains(byte)) {
                node->children[byte] = new TrieNode();
            }
            node = node->children[byte];
        }
        node->isToken = true;
    }
}

int Vocab::getVocabSize() const {
    return vocab.size();
}

const std::vector<std::string>& Vocab::getVocab() const {
    return vocab;
}

std::vector<std::string> Vocab::tokenize(const std::string& input) const {
    std::vector<uint8_t> bytes;

    for (size_t i = 0; i < input.size();) {
        unsigned char c = input[i];
        if (c == ' ') {
            bytes.insert(bytes.end(), MARKER.begin(), MARKER.end());
            i++;
        } else {
            size_t charLen = 1;
            if ((c & 0xF8) == 0xF0) charLen = 4;
            else if ((c & 0xF0) == 0xE0) charLen = 3;
            else if ((c & 0xE0) == 0xC0) charLen = 2;

            for (size_t j = 0; j < charLen; j++) bytes.push_back(input[i + j]);
            i += charLen;
        }
    }

    bytes.insert(bytes.begin(), MARKER.begin(), MARKER.end());

    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < bytes.size()) {
        TrieNode* node = trieRoot;
        int lastMatch = -1;
        size_t j = i;
        while (j < bytes.size() && node->children.contains(bytes[j])) {
            node =  node->children.at(bytes[j]);
            if (node->isToken) lastMatch = j;
            j++;
        }

        if (lastMatch != -1) {
            tokens.emplace_back(bytes.begin() + i, bytes.begin() + lastMatch + 1);
            i = lastMatch + 1;
        } else {
            tokens.emplace_back("<unk>");
            i++;
        }
    }
    return tokens;
}

std::vector<int> Vocab::encode(const std::string& text) const {
    std::vector<std::string> toks = tokenize(text);
    std::vector<int> ids;
    ids.reserve(toks.size());
    
    for (size_t i = 0; i < toks.size(); i++) {
        auto it = tokenToId.find(toks[i]);
        if (it != tokenToId.end()) {
            ids.push_back(it->second);
        } else {
            auto unkIt = tokenToId.find("<unk>");
            if (unkIt == tokenToId.end()) {
                throw std::runtime_error("Token <unk> not found in vocab");
            }
            ids.push_back(unkIt->second);
        }

        if (toks.size() > 0) {
            std::cout << "Encoding: " << (i * 100 / toks.size()) << "% done\n";
        }
    }
    return ids;
}
