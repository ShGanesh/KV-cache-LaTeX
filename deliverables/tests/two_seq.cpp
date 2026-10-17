// Two prompts in two sequences vs. separate runs.
//
// Joint run:    prompt A on seq 0 and prompt B on seq 1, decoded together (one batch per step).
// Separate run: A alone in a fresh cache, then B alone in a fresh cache.
// Pass if both prompts produce the same greedy tokens in both runs; also report the largest
// logit difference seen at any step, and the timings.
//
// usage: two_seq MODEL [n_gen=32] [kv_unified=1] [flash_attn=auto|on|off]
#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static llama_model * model;
static const llama_vocab * vocab;
static int n_vocab;
static llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;

static const char * PROMPTS[2] = {
    "\\begin{theorem}\\label{thm:qx}\nLet $f_{\\theta}$ be $\\lambda$-smooth with $\\lambda = 3.71$. Then",
    "\\section{Experiments}\nWe trained the critic on the Kestrel-9 dataset for 4100 iterations and observed that",
};

struct Run {
    std::vector<llama_token> tokens[2];  // generated tokens per prompt
    std::vector<float>       logits[2];  // logits of every step, concatenated
    double ms = 0;
};

static std::vector<llama_token> tokenize(const std::string & text) {
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), n, true, true);
    return toks;
}

static std::string detok(const std::vector<llama_token> & toks) {
    std::string s;
    char buf[128];
    for (auto t : toks) {
        int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        s.append(buf, n > 0 ? n : 0);
    }
    return s;
}

static void batch_add(llama_batch & b, llama_token tok, llama_pos pos, llama_seq_id seq, bool logits) {
    const int i = b.n_tokens++;
    b.token[i] = tok; b.pos[i] = pos; b.n_seq_id[i] = 1; b.seq_id[i][0] = seq; b.logits[i] = logits;
}

static llama_token argmax(const float * l) {
    int best = 0;
    for (int k = 1; k < n_vocab; ++k) if (l[k] > l[best]) best = k;
    return best;
}

// Greedy-generate n_gen tokens for each prompt in `which`, all in one context, one decode per step.
// Prompt which[j] goes into sequence j.
static Run generate(const std::vector<int> & which, int n_gen, bool kv_unified) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 512;
    cp.n_batch    = 512;
    cp.n_seq_max  = which.size();
    cp.kv_unified = kv_unified;
    cp.flash_attn_type = flash_attn;
    llama_context * ctx = llama_init_from_model(model, cp);

    Run run;
    llama_batch b = llama_batch_init(512, 0, 1);
    std::vector<llama_pos> pos(which.size());
    std::vector<int>       row(which.size());   // batch row holding seq j's logits

    const auto t0 = std::chrono::steady_clock::now();

    // prefill: every prompt in the same batch, each in its own sequence starting at position 0
    for (size_t j = 0; j < which.size(); ++j) {
        auto toks = tokenize(PROMPTS[which[j]]);
        for (size_t i = 0; i < toks.size(); ++i) batch_add(b, toks[i], i, j, i + 1 == toks.size());
        pos[j] = toks.size();
        row[j] = b.n_tokens - 1;
    }
    if (llama_decode(ctx, b)) { fprintf(stderr, "prefill failed\n"); exit(1); }

    // generation: one row per sequence per step
    for (int step = 0; step < n_gen; ++step) {
        std::vector<llama_token> next(which.size());
        for (size_t j = 0; j < which.size(); ++j) {
            const float * l = llama_get_logits_ith(ctx, row[j]);
            run.logits[which[j]].insert(run.logits[which[j]].end(), l, l + n_vocab);
            next[j] = argmax(l);
            run.tokens[which[j]].push_back(next[j]);
        }
        if (step + 1 == n_gen) break;
        b.n_tokens = 0;
        for (size_t j = 0; j < which.size(); ++j) {
            row[j] = b.n_tokens;
            batch_add(b, next[j], pos[j]++, j, true);
        }
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); exit(1); }
    }

    run.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    llama_batch_free(b);
    llama_free(ctx);
    return run;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [n_gen=32] [kv_unified=1] [flash_attn=auto|on|off]\n", argv[0]); return 2; }
    const int  n_gen      = argc > 2 ? atoi(argv[2]) : 32;
    const bool kv_unified = argc > 3 ? atoi(argv[3]) != 0 : true;
    const std::string fa  = argc > 4 ? argv[4] : "auto";
    flash_attn = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;

    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    llama_backend_init();
    model = llama_model_load_from_file(argv[1], llama_model_default_params());
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    vocab   = llama_model_get_vocab(model);
    n_vocab = llama_vocab_n_tokens(vocab);

    const Run joint = generate({0, 1}, n_gen, kv_unified);
    const Run solo0 = generate({0}, n_gen, kv_unified);
    const Run solo1 = generate({1}, n_gen, kv_unified);
    const Run * solo[2] = { &solo0, &solo1 };

    printf("n_gen = %d, kv_unified = %s, flash_attn = %s\n\n", n_gen, kv_unified ? "true" : "false", fa.c_str());
    bool pass = true;
    for (int p = 0; p < 2; ++p) {
        const auto & a = joint.tokens[p];
        const auto & s = solo[p]->tokens[p];
        int first_diff = -1;
        for (size_t i = 0; i < a.size(); ++i) if (a[i] != s[i]) { first_diff = i; break; }
        float maxd = 0;
        for (size_t k = 0; k < joint.logits[p].size(); ++k)
            maxd = std::max(maxd, std::fabs(joint.logits[p][k] - solo[p]->logits[p][k]));
        pass &= first_diff < 0;

        printf("prompt %d (seq %d in the joint run)\n", p, p);
        printf("  joint   : %s\n", detok(a).c_str());
        printf("  separate: %s\n", detok(s).c_str());
        printf("  tokens identical: %s", first_diff < 0 ? "YES" : "NO");
        if (first_diff >= 0) printf(" (first difference at token %d)", first_diff);
        printf("   max |logit diff| over %d steps: %g\n\n", n_gen, maxd);
    }
    printf("time: joint %.0f ms | separate %.0f + %.0f = %.0f ms\n", joint.ms, solo0.ms, solo1.ms, solo0.ms + solo1.ms);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    llama_model_free(model);
    llama_backend_free();
    return pass ? 0 : 1;
}
