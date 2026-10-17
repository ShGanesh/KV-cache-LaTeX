// Structured-sequence driver, based on llama.cpp/examples/batched/batched.cpp.
//
// Loads a parsed paper (deliverables/schema/schema.json), prefills it with one sequence per group, answers a question
// in one group, and optionally
//   --replace NODE TEXT : change one leaf's text and recompute only its group
//   --evict SEQ         : remove a group's cells by position range
// After each operation the question is asked again and compared with the first answer.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

using json = nlohmann::json;

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf --doc data/json/1904.09237.json --seq 1 -p \"What is the caption of Algorithm 1?\" -n 32 [--replace NODE TEXT] [--evict SEQ]\n", argv[0]);
    LOG("\n");
}

// Append the question to `seq`, greedy-decode up to n_predict tokens (stop at a newline), then
// remove question + answer again so the cache is back to its prefill state.
static std::vector<llama_token> ask(llama_context * ctx, llama_sampler * smpl, common_batch & batch, llama_seq_id seq,
                                    const std::string & question, int n_predict, double & ms) {
    const auto t_start = ggml_time_us();
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    llama_memory_t mem = llama_get_memory(ctx);

    const llama_pos start = llama_memory_seq_pos_max(mem, seq) + 1;
    const auto q = common_tokenize(vocab, "\n\nQuestion: " + question + "\nAnswer:", false);

    batch.clear();
    for (size_t i = 0; i < q.size(); ++i) {
        batch.add(q[i], start + i, seq, false);
    }
    // llama_decode will output logits only for the last token of the question
    batch.set_output(batch.size() - 1, true);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        exit(1);
    }

    std::vector<llama_token> answer;
    int n_cur = start + q.size();
    int i_batch = batch.size() - 1;
    for (int n = 0; n < n_predict; ++n) {
        const llama_token new_token_id = llama_sampler_sample(smpl, ctx, i_batch);
        if (llama_vocab_is_eog(vocab, new_token_id) || common_token_to_piece(ctx, new_token_id).find('\n') != std::string::npos) {
            break;
        }
        answer.push_back(new_token_id);

        batch.clear();
        i_batch = batch.add(new_token_id, n_cur++, seq, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("%s : failed to eval\n", __func__);
            exit(1);
        }
    }

    llama_memory_seq_rm(mem, seq, start, -1);
    ms = (ggml_time_us() - t_start) / 1000.0;
    return answer;
}

static std::string to_text(llama_context * ctx, const std::vector<llama_token> & toks) {
    std::string s;
    for (auto t : toks) {
        s += common_token_to_piece(ctx, t);
    }
    return s;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // our own arguments; everything else (-m, -p, -n, -t, ...) goes to common_params_parse
    std::string doc_path, replace_text;
    int seq = -1, replace_node = -1, evict_seq = -1;
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--doc"     && i + 1 < argc) { doc_path = argv[++i]; }
        else if (a == "--seq"     && i + 1 < argc) { seq = std::stoi(argv[++i]); }
        else if (a == "--evict"   && i + 1 < argc) { evict_seq = std::stoi(argv[++i]); }
        else if (a == "--replace" && i + 2 < argc) { replace_node = std::stoi(argv[++i]); replace_text = argv[++i]; }
        else { rest.push_back(argv[i]); }
    }

    common_params params;

    params.prompt = "";
    params.n_predict = 32;

    common_init();

    if (!common_params_parse(rest.size(), rest.data(), params, LLAMA_EXAMPLE_BATCHED, print_usage)) {
        return 1;
    }

    // the question
    const std::string question = params.prompt;

    // maximum number of answer tokens
    int n_predict = params.n_predict;

    // load the paper
    json doc;
    std::ifstream(doc_path) >> doc;
    if (doc.is_null() || question.empty()) {
        print_usage(argc, argv);
        return 1;
    }
    auto & nodes = doc["nodes"];
    const int n_seq = doc["n_seq"];
    const int G = n_seq - 1;
    if (seq < 1 || seq > G || (evict_seq != -1 && (evict_seq < 1 || evict_seq > G || evict_seq == seq))) {
        LOG_ERR("%s: --seq must be 1..%d, and --evict must be another group\n", __func__, G);
        return 1;
    }
    const json * root = nullptr;
    std::vector<std::vector<json *>> own(G + 1), shared_in(G + 1);   // per group: own leaves, leaves shared in
    for (auto & n : nodes) {
        if (n["tokens"].empty()) {
            continue;
        }
        if (n["node_type"] == "root") {
            root = &n;
            continue;
        }
        const auto & sids = n["seq_ids"];
        own[sids[0].get<int>()].push_back(&n);
        for (size_t k = 1; k < sids.size(); ++k) {
            shared_in[sids[k].get<int>()].push_back(&n);
        }
    }

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = common_model_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // initialize the context

    llama_context_params ctx_params = common_context_params_to_llama(params);

    ctx_params.n_ctx           = 4096;
    ctx_params.n_batch         = 2048;
    ctx_params.n_seq_max       = n_seq;
    ctx_params.kv_unified      = true;                           // the root's cells belong to every seq
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // bit-identical outputs (see deliverables/tests/two_seq.cpp)
    // one thread for single-token decodes: with several, freeing cells elsewhere in the cache changed the rounding
    // of the answer (measured; our reading is that the work is split across threads by cell index). Batches
    // (prefill, question) keep -tb.
    ctx_params.n_threads       = 1;

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        LOG_ERR("%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    // create a llama_batch
    // we use this object to submit token data for decoding
    common_batch batch(ctx);

    // prefill, in the decode order of deliverables/schema/MERGE_RULES.md

    const auto t_prefill_start = ggml_time_us();

    // the root is shared by every sequence
    std::vector<llama_seq_id> seq_ids(n_seq, 0);
    for (int32_t i = 0; i < n_seq; ++i) {
        seq_ids[i] = i;
    }

    const auto root_tokens = (*root)["tokens"].get<std::vector<llama_token>>();
    batch.clear();
    for (size_t i = 0; i < root_tokens.size(); ++i) {
        batch.add(root_tokens[i], i, seq_ids, false);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return 1;
    }
    int n_prefill = root_tokens.size();

    // group order: the question's group first (after the groups it cites), then the rest in document order.
    // Every other group's cells then lie after the question group's, so replacing or evicting one never moves
    // the question in front of them in the cache.
    std::vector<int> order;
    std::vector<bool> placed(G + 1, false);
    std::function<void(int)> place = [&](int g) {
        if (placed[g]) {
            return;
        }
        placed[g] = true;
        for (auto * n : shared_in[g]) {
            place((*n)["seq_ids"][0].get<int>());   // cited groups are decoded before the citing group
        }
        order.push_back(g);
    };
    place(seq);
    for (int g = 1; g <= G; ++g) {
        place(g);
    }

    // each group: first share in the leaves it cites, then decode its own leaves
    for (int g : order) {
        for (auto * n : shared_in[g]) {
            const auto & pos = (*n)["positions"];
            llama_memory_seq_cp(mem, (*n)["seq_ids"][0].get<int>(), g, pos.front().get<int>(), pos.back().get<int>() + 1);
        }
        batch.clear();
        for (auto * n : own[g]) {
            const auto toks = (*n)["tokens"].get<std::vector<llama_token>>();
            const llama_pos p0 = (*n)["positions"][0];
            for (size_t i = 0; i < toks.size(); ++i) {
                batch.add(toks[i], p0 + i, g, false);
            }
        }
        n_prefill += batch.size();
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }

    const double prefill_ms = (ggml_time_us() - t_prefill_start) / 1000.0;

    LOG("\ndoc %s: %d tokens, %d groups, prefill %.0f ms (%.1f t/s)\n",
        doc["doc_id"].get<std::string>().c_str(), n_prefill, G, prefill_ms, n_prefill / prefill_ms * 1000);
    for (int g = 1; g <= G; ++g) {
        LOG("  seq %d: positions %d..%d\n", g, llama_memory_seq_pos_min(mem, g), llama_memory_seq_pos_max(mem, g));
    }

    json result = {{"doc_id", doc["doc_id"]}, {"seq", seq}, {"question", question},
                   {"n_tokens", n_prefill}, {"prefill_ms", prefill_ms}};

    // baseline answer

    double ms;
    const auto base = ask(ctx, smpl, batch, seq, question, n_predict, ms);
    LOG("\nQ (seq %d): %s\nA: %s   [%zu tokens, %.0f ms]\n", seq, question.c_str(), to_text(ctx, base).c_str(), base.size(), ms);
    result["answer"]     = to_text(ctx, base);
    result["answer_ids"] = base;
    result["answer_ms"]  = ms;

    // change one leaf, recompute only its group

    if (replace_node >= 0) {
        auto & target = nodes.at(replace_node);
        if (target["tokens"].empty() || target["node_type"] == "root" || target["seq_ids"][0] == seq) {
            LOG_ERR("%s: --replace needs a non-root leaf outside seq %d\n", __func__, seq);
            return 1;
        }
        const int b = target["seq_ids"][0];
        if (target["seq_ids"].size() > 1) {
            LOG_WRN("%s: node %d is shared into other seqs; they keep the old text\n", __func__, replace_node);
        }

        const auto t_start = ggml_time_us();
        const llama_pos p0 = (*own[b].front())["positions"][0];
        llama_memory_seq_rm(mem, b, p0, -1);                      // drop group b's own cells
        batch.clear();
        llama_pos p = p0;
        for (auto * n : own[b]) {
            const auto toks = (n == &target) ? common_tokenize(vocab, replace_text, false) : (*n)["tokens"].get<std::vector<llama_token>>();
            for (auto t : toks) {
                batch.add(t, p++, b, false);
            }
        }
        const int n_recomputed = batch.size();
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {   // re-decode only group b
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return 1;
        }
        const double replace_ms = (ggml_time_us() - t_start) / 1000.0;

        const auto after = ask(ctx, smpl, batch, seq, question, n_predict, ms);
        LOG("\nreplaced node %d (seq %d): recomputed %d tokens in %.0f ms\nA: %s   identical: %s\n",
            replace_node, b, n_recomputed, replace_ms, to_text(ctx, after).c_str(), after == base ? "YES" : "NO");
        result["replace"] = {{"node", replace_node}, {"seq", b}, {"recomputed_tokens", n_recomputed}, {"ms", replace_ms},
                             {"answer", to_text(ctx, after)}, {"identical", after == base}};
    }

    // evict a group by position range

    if (evict_seq >= 0) {
        const llama_pos p0 = (*own[evict_seq].front())["positions"][0];
        const llama_pos p1 = llama_memory_seq_pos_max(mem, evict_seq) + 1;
        const size_t kv0 = llama_state_get_size(ctx);             // serializes only used cells
        llama_memory_seq_rm(mem, evict_seq, p0, p1);
        const size_t kv1 = llama_state_get_size(ctx);

        const auto after = ask(ctx, smpl, batch, seq, question, n_predict, ms);
        LOG("\nevicted seq %d, positions [%d, %d): KV in use %.1f -> %.1f KiB (-%.1f KiB)\nA: %s   identical: %s\n",
            evict_seq, p0, p1, kv0 / 1024.0, kv1 / 1024.0, (kv0 - kv1) / 1024.0, to_text(ctx, after).c_str(), after == base ? "YES" : "NO");
        result["evict"] = {{"seq", evict_seq}, {"p0", p0}, {"p1", p1}, {"kv_bytes_before", kv0}, {"kv_bytes_after", kv1},
                           {"answer", to_text(ctx, after)}, {"identical", after == base}};
    }

    LOG("\nRESULT %s\n", result.dump().c_str());

    LOG("\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);

    fprintf(stderr, "\n");

    llama_sampler_free(smpl);

    llama_free(ctx);
    llama_model_free(model);

    llama_backend_free();

    return 0;
}
