
#include "suffix_tree.h"
#include <torch/extension.h>
#include <vector>

// Wrapper for Draft to be exposed to Python via torch::custom_class
struct SuffixDecodingDraft : torch::CustomClassHolder {
    Draft draft;
    SuffixDecodingDraft() = default;
    SuffixDecodingDraft(Draft d) : draft(std::move(d)) {}
    
    std::vector<int64_t> get_token_ids() const {
        return std::vector<int64_t>(draft.token_ids.begin(), draft.token_ids.end());
    }
    std::vector<int64_t> get_parents() const {
        return std::vector<int64_t>(draft.parents.begin(), draft.parents.end());
    }
    std::vector<double> get_probs() const {
        return std::vector<double>(draft.probs.begin(), draft.probs.end());
    }
    double get_score() const { return draft.score; }
    int64_t get_match_len() const { return draft.match_len; }
};

struct SuffixDecodingTree : torch::CustomClassHolder {
    SuffixTree tree;
    SuffixDecodingTree(int64_t max_depth) : tree(static_cast<int>(max_depth)) {}

    int64_t num_seqs() const { return tree.num_seqs(); }

    void remove(int64_t seq_id) { tree.remove(static_cast<int>(seq_id)); }

    void extend(int64_t seq_id, torch::Tensor tokens) {
        // Ensure CPU and contiguous
        tokens = tokens.cpu().contiguous();
        
        if (tokens.scalar_type() == torch::kInt32) {
            Span<const int32_t> s(tokens.data_ptr<int32_t>(), tokens.numel());
            tree.extend(static_cast<int>(seq_id), s);
        } else if (tokens.scalar_type() == torch::kInt64) {
            // Convert to int32
            auto tokens_i32 = tokens.to(torch::kInt32);
            Span<const int32_t> s(tokens_i32.data_ptr<int32_t>(), tokens_i32.numel());
            tree.extend(static_cast<int>(seq_id), s);
        } else {
            throw std::runtime_error("extend expects int32 or int64 tensor");
        }
    }

    c10::intrusive_ptr<SuffixDecodingDraft> speculate(
        torch::Tensor context,
        int64_t max_spec_tokens,
        double max_spec_factor,
        double max_spec_offset,
        double min_token_prob,
        bool use_tree_spec
    ) {
        context = context.cpu().contiguous();
        std::vector<int32_t> context_vec;
        if (context.scalar_type() == torch::kInt32) {
             context_vec.assign(context.data_ptr<int32_t>(), context.data_ptr<int32_t>() + context.numel());
        } else {
             auto c32 = context.to(torch::kInt32);
             context_vec.assign(c32.data_ptr<int32_t>(), c32.data_ptr<int32_t>() + c32.numel());
        }
        Span<const int32_t> context_span(context_vec.data(), context_vec.size());
        
        Draft d = tree.speculate(
            context_span,
            static_cast<int>(max_spec_tokens),
            static_cast<float>(max_spec_factor),
            static_cast<float>(max_spec_offset),
            static_cast<float>(min_token_prob),
            use_tree_spec
        );
        return c10::make_intrusive<SuffixDecodingDraft>(std::move(d));
    }

    std::string check_integrity() { return tree.check_integrity(); }
    int64_t estimate_memory() const { return static_cast<int64_t>(tree.estimate_memory()); }
};

// Register via Fragment to add to the existing vllm library
TORCH_LIBRARY_FRAGMENT(vllm, m) {
    m.class_<SuffixDecodingDraft>("SuffixDecodingDraft")
        .def_property("token_ids", &SuffixDecodingDraft::get_token_ids)
        .def_property("parents", &SuffixDecodingDraft::get_parents)
        .def_property("probs", &SuffixDecodingDraft::get_probs)
        .def_property("score", &SuffixDecodingDraft::get_score)
        .def_property("match_len", &SuffixDecodingDraft::get_match_len);

    m.class_<SuffixDecodingTree>("SuffixDecodingTree")
        .def(torch::init<int64_t>()) // max_depth
        .def("num_seqs", &SuffixDecodingTree::num_seqs)
        .def("remove", &SuffixDecodingTree::remove)
        .def("extend", &SuffixDecodingTree::extend)
        .def("speculate", &SuffixDecodingTree::speculate)
        .def("check_integrity", &SuffixDecodingTree::check_integrity)
        .def("estimate_memory", &SuffixDecodingTree::estimate_memory);
}
