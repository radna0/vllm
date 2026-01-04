
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp algorithm to project a matrix onto the Birkhoff polytope
    (doubly stochastic matrices).
    """
    def __init__(self, num_iters: int = 3, epsilon: float = 1e-8):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    def forward(self, log_alpha: torch.Tensor) -> torch.Tensor:
        # Input log_alpha is (Batch, n, n) or (n, n)
        # We assume input is already in log-space or logits.
        # Paper suggests starting with exp(log_alpha).
        # To avoid overflow, we can work in log-space or simply expoentiate if stable.
        # Given n is small (e.g. 4), direct exp is usually fine.
        
        # P = exp(Q)
        P = torch.exp(log_alpha)
        
        for _ in range(self.num_iters):
            # Row normalization
            P = P / (P.sum(dim=-1, keepdim=True) + self.epsilon)
            # Column normalization
            P = P / (P.sum(dim=-2, keepdim=True) + self.epsilon)
            
        return P

class MHCAdapter(nn.Module):
    """
    Multi-Head Connection Adapter.
    Expands the residual stream from C to n*C.
    Implements: X_{l+1} = H^{res}_l X_l + H^{post \top}_l F(H^{pre}_l X_l, W_l)
    """
    def __init__(self, hidden_size: int, num_heads: int = 4, init_identity: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Learnable parameters
        # H_res: (n, n) mixing matrix (parameterized as logits for Sinkhorn)
        self.h_res_logits = nn.Parameter(torch.randn(num_heads, num_heads))
        
        # H_pre: (n,) weighing vector (parameterized to be passed through sigmoid)
        self.h_pre_logits = nn.Parameter(torch.zeros(num_heads))
        
        # H_post: (n,) weighing vector (parameterized to be passed through 2*sigmoid)
        self.h_post_logits = nn.Parameter(torch.zeros(num_heads))
        
        self.sinkhorn = SinkhornKnopp()
        
        if init_identity:
            self.initialize_identity()

    def initialize_identity(self):
        """
        Initialize parameters such that:
        - H_res is Identity
        - H_pre selects stream 0
        - H_post writes to stream 0
        """
        # 1. H_pre -> [1, 0, 0, ...]
        # Sigmoid(x). large positive for idx 0, large negative for others.
        with torch.no_grad():
            self.h_pre_logits.data.fill_(-10.0)
            self.h_pre_logits.data[0] = 10.0
            
            # 2. H_post -> [1, 0, 0, ...]
            # 2 * Sigmoid(x). 
            # We want approx 1.0 for idx 0 -> Sigmoid should be 0.5 -> x=0.0
            # Wait, 2 * Sigmoid(0) = 2 * 0.5 = 1.0. Correct.
            # We want 0.0 for others -> Sigmoid ~ 0 -> x = -large
            self.h_post_logits.data.fill_(-10.0)
            self.h_post_logits.data[0] = 0.0 
            
            # 3. H_res -> Identity
            # Sinkhorn projects exp(logits).
            # We want exp(logits) to be diagonal dominant.
            self.h_res_logits.data.fill_(-10.0)
            self.h_res_logits.data.diagonal().fill_(10.0)

    def get_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H_res = self.sinkhorn(self.h_res_logits) # (n, n)
        H_pre = torch.sigmoid(self.h_pre_logits) # (n,)
        H_post = 2 * torch.sigmoid(self.h_post_logits) # (n,)
        return H_res, H_pre, H_post

    def pre_process(self, x_streams: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_streams: (Batch, Seq, n, C)
        Returns:
            x_in: (Batch, Seq, C) - input to the block
            x_mixed: (Batch, Seq, n, C) - mixed residual state
        """
        H_res, H_pre, _ = self.get_matrices()
        
        # 1. Mix streams: X_mixed = H_res * X
        # H_res is (n, n), X is (..., n, C).
        # Einsum: ij, ...jk -> ...ik
        x_mixed = torch.einsum('ij, ...jk -> ...ik', H_res, x_streams)
        
        # 2. Project to input: x_in = H_pre . X
        # H_pre is (n,), X is (..., n, C)
        # We broadcast H_pre over Batch, Seq, C.
        # Easier: weighted sum over dimension n.
        x_in = torch.einsum('j, ...jk -> ...k', H_pre, x_streams)
        
        return x_in, x_mixed

    def post_process(self, x_mixed: torch.Tensor, block_output: torch.Tensor) -> torch.Tensor:
        """
        x_mixed: (Batch, Seq, n, C)
        block_output: (Batch, Seq, C)
        Returns:
            x_next: (Batch, Seq, n, C)
        """
        _, _, H_post = self.get_matrices()
        
        # X_{l+1} = X_mixed + H_post^T * block_output
        # H_post is (n,), block_output is (..., C).
        # Outer product to get (..., n, C)
        update_term = torch.einsum('i, ...k -> ...ik', H_post, block_output)
        
        x_next = x_mixed + update_term
        return x_next
