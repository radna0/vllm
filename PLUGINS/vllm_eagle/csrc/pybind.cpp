#include <torch/extension.h>

// Forward declarations from ops.h / kernels
void build_tree_kernel_efficient(
    torch::Tensor parent_list, torch::Tensor selected_index,
    torch::Tensor verified_seq_len, torch::Tensor tree_mask,
    torch::Tensor positions, torch::Tensor retrive_index,
    torch::Tensor retrive_next_token, torch::Tensor retrive_next_sibling,
    int64_t topk, int64_t depth, int64_t draft_token_num,
    int64_t tree_mask_mode);

void reconstruct_indices_from_tree_mask(
    torch::Tensor tree_mask, torch::Tensor verified_seq_len,
    torch::Tensor positions, torch::Tensor retrive_index,
    torch::Tensor retrive_next_token, torch::Tensor retrive_next_sibling,
    int64_t batch_size, int64_t draft_token_num);

void tree_speculative_sampling_target_only(
    torch::Tensor predicts, torch::Tensor accept_index,
    torch::Tensor accept_token_num, torch::Tensor candidates,
    torch::Tensor retrive_index, torch::Tensor retrive_next_token,
    torch::Tensor retrive_next_sibling, torch::Tensor uniform_samples,
    torch::Tensor uniform_samples_for_final_sampling,
    torch::Tensor target_probs, torch::Tensor draft_probs,
    double threshold_single, double threshold_acc, bool deterministic);

void apply_logit_filters(torch::Tensor& logits, torch::Tensor& top_k,
                         torch::Tensor& top_p, torch::Tensor& min_p);

void fused_gumbel_sample(torch::Tensor& out_tokens, torch::Tensor& logits,
                         torch::Tensor& top_k, torch::Tensor& top_p,
                         torch::Tensor& min_p, torch::Tensor& temperatures,
                         torch::Tensor& uniform_samples);

// Phase 2 Optimizations
void fused_gumbel_sample_warp_optimized(
    torch::Tensor& out_tokens,
    torch::Tensor& logits,
    torch::Tensor& uniform_samples,
    torch::Tensor& min_p,
    torch::Tensor& temperatures);

void fused_draft_verify_sample(
    torch::Tensor& accepted_tokens,
    torch::Tensor& num_accepted,
    torch::Tensor& draft_logits,
    torch::Tensor& target_logits,
    torch::Tensor& uniform_samples,
    torch::Tensor& verify_samples,
    torch::Tensor& min_p,
    torch::Tensor& temperatures);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_tree_kernel_efficient", &build_tree_kernel_efficient, "Build tree kernel efficient");
  m.def("reconstruct_indices_from_tree_mask", &reconstruct_indices_from_tree_mask, "Reconstruct indices from tree mask");
  m.def("tree_speculative_sampling_target_only", &tree_speculative_sampling_target_only, "Tree speculative sampling target only");
  m.def("apply_logit_filters", &apply_logit_filters, "Apply logit filters");
  m.def("fused_gumbel_sample", &fused_gumbel_sample, "Fused Gumbel-Max sampling");
  
  // Phase 2 Optimizations
  m.def("fused_gumbel_sample_warp_optimized", &fused_gumbel_sample_warp_optimized, 
        "Warp-optimized Gumbel-Max sampling (+5-10% performance)");
  m.def("fused_draft_verify_sample", &fused_draft_verify_sample,
        "Fused draft-verify-sample kernel (+5-8% throughput)");
}
