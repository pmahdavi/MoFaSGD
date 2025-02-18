import torch
from torch.optim import Optimizer

#############################################
#    MomentumFactor helper class
#############################################
class MomentumFactor:
    """
    Stores a rank-r factorization (U, S, V) for the momentum matrix:
      M ≈ U diag(S) V^T,  with U in R^(m x r), S in R^r, V in R^(n x r).
    Initialization uses a truncated SVD of the parameter p.
    """
    def __init__(self, p: torch.Tensor, rank: int):
        """
        Args:
          p:    (m x n) parameter tensor
          rank: desired rank r
        """
        m, n = p.shape
        # Full SVD (may be expensive for large matrices, but fine for demonstration)
        U_full, S_full, V_full = torch.svd(p)
        r_trunc = min(rank, min(m, n))
        self.U = U_full[:, :r_trunc].clone()
        self.S = S_full[:r_trunc].clone()
        self.V = V_full[:, :r_trunc].clone()


#############################################
#   The MomentumFactorizedSGD Optimizer
#############################################
class MomentumFactorizedSGD(Optimizer):
    r"""
    Implements a rank-r factorized momentum update:
    
      1) We store M_t = U_t diag(S_t) V_t^T  (rank r).
      2) Each step, we do:

         (a) Tangent project the new gradient G_t:
             G_hat = U_t U_t^T G_t
                     + G_t V_t V_t^T
                     - U_t U_t^T G_t V_t V_t^T.

         (b) Update the momentum factor to represent:
             M_t = G_hat + beta * M_{t-1} 
                  = G_hat + beta * (U_t diag(S_t) V_t^T)
             with rank at most 2r, truncated back to r via the
             block-matrix approach:

             [U_t | G_t V_t] = U'_t R_{U_t}, 
             [V_t | G_t^T U_t] = V'_t R_{V_t},     # QR factorizations

             B = [[ beta diag(S_t) - U_t^T G_t V_t,   I_r ],
                  [             I_r,                 0_r ]]

             Mid = R_{U_t} * B * R_{V_t}^T
             => SVD_r(Mid) = U'' diag(S'') V''^T
             => U_{t+1} = U'_t U'',
                S_{t+1} = S'',
                V_{t+1} = V'_t V''.

         (c) Finally, update p:
             p <- p - lr * [
                  eta1 * (U_{t+1} diag(1/S_{t+1}) V_{t+1}^T)
                  + eta2 * (I - P_U) G_t (I - P_V)
             ].

    Args:
      params: iterable of parameters to optimize (all must be 2D)
      lr:     global learning rate
      rank:   integer rank r
      beta:   momentum decay factor
      eta1:   scale factor for the low-rank momentum term
      eta2:   scale factor for the orthogonal complement gradient
      use_current_projection: flag to use current or previous projections
      use_ones_for_nonzero_s: flag to handle singular values
    """

    def __init__(self, params,
                 lr=1e-2,
                 rank=2,
                 beta=0.9,
                 eta1=1.0,
                 eta2=1.0,
                 use_current_projection=False,
                 use_ones_for_nonzero_s=False):
        defaults = dict(lr=lr, rank=rank, beta=beta, eta1=eta1, eta2=eta2,
                       use_current_projection=use_current_projection,
                       use_ones_for_nonzero_s=use_ones_for_nonzero_s)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group['lr']
            rank = group['rank']
            beta = group['beta']
            eta1 = group['eta1']
            eta2 = group['eta2']
            use_current_projection = group['use_current_projection']
            use_ones_for_nonzero_s = group['use_ones_for_nonzero_s']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Check dimension => Must be 2D
                if p.dim() != 2:
                    raise RuntimeError(
                        "MomentumFactorizedSGD only supports 2D parameters."
                    )

                # Retrieve or init (U, S, V)
                state = self.state[p]
                if 'momentum_factor' not in state:
                    state['momentum_factor'] = MomentumFactor(p, rank)

                mf = state['momentum_factor']
                U, S, V = mf.U, mf.S, mf.V

                # Current gradient
                G_t = p.grad

                # == (a) Tangent Projection ==
                UUTG = U @ (U.T @ G_t)            # (m x n)
                GVVT = (G_t @ V) @ V.T            # (m x n)
                G_hat = UUTG + GVVT - (UUTG @ V) @ V.T  # shape (m x n)

                # == (b) Update the momentum factor ==
                # We want M_new = G_hat + beta * (U diag(S) V^T),
                # and store it as U_{t+1} diag(S_{t+1}) V_{t+1}^T

                U_next, S_next, V_next = self._update_momentum_factor(U, S, V, G_hat, beta)

                # Store the new factors
                mf.U = U_next
                mf.S = S_next
                mf.V = V_next

                # == (c) Parameter update ==
                #    p <- p - lr * [
                #        eta1 * (U_{t+1} diag(1/S_{t+1}) V_{t+1}^T)
                #        + eta2 * (I - P_U) G_t (I - P_V)
                #    ]
                #    where P_U, P_V are either current or previous projections
                #    based on use_current_projection flag
                U_nextU_nextT = U_next @ U_next.T  # (m x m)
                V_nextV_nextT = V_next @ V_next.T  # (n x n)

                # Add numerical stability for division
                eps = 1e-3
                # Create mask for non-zero singular values
                non_zero_mask = S_next.abs() > eps
                safe_reciprocal_S = torch.zeros_like(S_next)
                
                # Handle singular values based on use_ones_for_nonzero_s flag
                if group['use_ones_for_nonzero_s']:
                    # Set all non-zero singular values to 1.0 (equivalent to setting their reciprocal to 1.0)
                    safe_reciprocal_S[non_zero_mask] = 1.0
                else:
                    # Original behavior: compute reciprocal for non-zero values
                    safe_reciprocal_S[non_zero_mask] = 1.0 / (S_next[non_zero_mask])
                    # Optionally clip extremely large values
                    max_value = 10000
                    safe_reciprocal_S = torch.clamp(safe_reciprocal_S, max=max_value)

                # Low-rank momentum part with reciprocal S
                USVt_next = (U_next * safe_reciprocal_S.unsqueeze(0)) @ V_next.T  # shape (m x n)

                # Choose projection matrices based on flag
                if use_current_projection:
                    proj_U = U_nextU_nextT
                    proj_V = V_nextV_nextT
                else:
                    proj_U = U @ U.T
                    proj_V = V @ V.T

                # Orthogonal complement of G_t
                left_ortho = G_t - proj_U @ G_t
                right_ortho = left_ortho - left_ortho @ proj_V

                # Final update
                p.data = p.data - lr * (
                    eta1 * USVt_next
                    + eta2 * right_ortho
                )

    #############################################
    # The core rank-2r update of momentum
    #############################################
    def _update_momentum_factor(self, U, S, V, G, beta):
        """
        Internal method to update (U, S, V) so that
           M_new = G + beta * (U diag(S) V^T)
        is represented by rank-r factors (U', S', V'),
        using the 2-QR + block-matrix + 2r×2r SVD approach.

        Equations match:

          [ U | G V ] = U'  R_U
          [ V | G^T U ] = V' R_V

          B = [[ beta diag(S) - U^T G V,  I_r ],
               [          I_r,           0_r ]]

          Mid = R_U * B * R_V^T   =>  SVD_r(Mid) = U'' diag(S'') V''^T

          =>  U_next = U' * U'',  S_next = S'',  V_next = V' * V''
        """
        m, r = U.shape
        n = V.shape[0]
        assert V.shape[1] == r, "V must be (n x r)"
        assert S.shape[0] == r, "S must be (r,)"
        assert G.shape == (m, n), "G must be (m x n)"

        # 1) row_block = [ U,  G V ]
        GV = G.mm(V)  # (m x r)
        row_block = torch.cat([U, GV], dim=1)  # (m, 2r)
        U_prime, R_U = torch.linalg.qr(row_block, mode='reduced')  # U_prime:(m,2r), R_U:(2r,2r)

        # 2) col_block = [ V,  G^T U ]
        GTU = G.t().mm(U)  # (n, r)
        col_block = torch.cat([V, GTU], dim=1)  # (n, 2r)
        V_prime, R_V = torch.linalg.qr(col_block, mode='reduced')  # V_prime:(n,2r), R_V:(2r,2r)

        # 3) B = block matrix of size (2r,2r)
        beta_Sigma = torch.diag(beta * S)           # (r x r)
        UTGV = U.t().mm(G).mm(V)                    # (r x r)
        top_left = beta_Sigma - UTGV                # (r x r)
        eye_r = torch.eye(r, device=U.device)
        zero_r = torch.zeros(r, r, device=U.device)

        top_row = torch.cat([top_left, eye_r], dim=1)   # (r,2r)
        bot_row = torch.cat([eye_r, zero_r], dim=1)     # (r,2r)
        B = torch.cat([top_row, bot_row], dim=0)        # (2r,2r)

        # 4) Mid = R_U * B * R_V^T
        Mid = R_U.mm(B).mm(R_V.t())  # (2r, 2r)

        # 5) SVD on Mid, truncated to rank r
        U_dblprime, S_dblprime, V_dblprime = torch.svd(Mid)
        U_dblprime_r = U_dblprime[:, :r]  # (2r, r)
        S_dblprime_r = S_dblprime[:r]     # (r,)
        V_dblprime_r = V_dblprime[:, :r]  # (2r, r)

        # print("\n=== SVD Debug Info ===")
        # print(f"Shape of Mid matrix: {Mid.shape}")
        # print(f"Rank parameter r: {r}")
        # print("Singular values summary:")
        
        # # Get total number of singular values
        # n_vals = len(S_dblprime_r)
        # # Show at most 3 values from start and end
        # n_show = min(3, n_vals // 2)
        
        # # Print first few values
        # print("First singular values:")
        # for i in range(min(n_show, n_vals)):
        #     print(f"  σ_{i+1}: {S_dblprime_r[i]:.6f}")
            
        # # If there are more values in the middle, show ellipsis
        # if n_vals > 2 * n_show:
        #     print("  ...")
            
        # # Print last few values if we have more
        # if n_vals > n_show:
        #     print("Last singular values:")
        #     for i in range(max(n_show, n_vals - n_show), n_vals):
        #         print(f"  σ_{i+1}: {S_dblprime_r[i]:.6f}")
                
        # # Print some statistics
        # print(f"Statistics:")
        # print(f"  Max σ: {torch.max(S_dblprime_r):.6f}")
        # print(f"  Min σ: {torch.min(S_dblprime_r):.6f}")
        # print(f"  Mean σ: {torch.mean(S_dblprime_r):.6f}")
        # print(f"  Median σ: {torch.median(S_dblprime_r):.6f}")
        # print("=====================\n")
        
        # 6) Pull back => U_next, S_next, V_next
        U_next = U_prime.mm(U_dblprime_r)  # (m, r)
        S_next = S_dblprime_r.clone()      # (r,)
        V_next = V_prime.mm(V_dblprime_r)  # (n, r)

        return U_next, S_next, V_next


# # ----------------------------------------------------------------------
# # Example usage:
# if __name__ == "__main__":
#     # Suppose a single 2D weight
#     w = torch.nn.Parameter(torch.randn(8, 6, requires_grad=True))

#     # Dummy closure/ loss
#     def loss_fn():
#         # e.g., sum of w for demonstration
#         return w.sum()

#     # Instantiate the optimizer
#     optim = MomentumFactorizedSGD([w], lr=1e-2, rank=2, beta=0.9, eta1=1.0, eta2=1.0)

#     # Simple loop
#     for step in range(5):
#         optim.zero_grad()
#         l = loss_fn()
#         l.backward()
#         optim.step()
#         print(f"Step {step}: loss={l.item():.4f}")