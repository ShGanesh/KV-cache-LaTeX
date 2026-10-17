// Tokenizer for the parser: loads the model's vocabulary once, tokenizes many texts.
//
// stdin : one JSON object per line, {"text": "...", "bos": true|false}
// stdout: one JSON object per line, {"ids": [...], "lens": [...]}
//         lens[i] = number of UTF-8 bytes of the text that token i covers (0 for BOS)
//
// usage: tokenize MODEL < texts.jsonl
#include "llama.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL < texts.jsonl\n", argv[0]); return 2; }

    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        const auto in = nlohmann::json::parse(line);
        const std::string text = in.at("text");
        const bool bos = in.value("bos", false);

        // parse_special = false: "<|eot_id|>" written inside a paper stays plain text
        const int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, bos, false);
        std::vector<llama_token> ids(n);
        llama_tokenize(vocab, text.c_str(), text.size(), ids.data(), n, bos, false);

        std::vector<int> lens(n);
        char buf[256];
        for (int i = 0; i < n; ++i) {
            // special = false: BOS renders as "" and so covers 0 bytes of the text
            lens[i] = std::max(0, llama_token_to_piece(vocab, ids[i], buf, sizeof(buf), 0, false));
        }
        std::cout << nlohmann::json{{"ids", ids}, {"lens", lens}}.dump() << "\n";
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
