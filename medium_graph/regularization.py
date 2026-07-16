import torch

def estimate_cooccurrence_matrix(predictions, edge_index, num_classes, device):
    """
    Estimates the global co-occurrence matrix from current model predictions.
    
    Args:
        predictions (Tensor): The predicted probabilities (N x C).
        edge_index (Tensor): The graph edge indices (2 x E).
        num_classes (int): Number of classes.
        device (torch.device): Device to compute on.
        
    Returns:
        Tensor: A C x C normalized co-occurrence matrix.
    """
    src, dst = edge_index
    
    # predictions[src]^T @ predictions[dst] yields a (C x C) matrix
    # where entry (i, j) is the sum over all edges (u,v) of P(Y_u=i) * P(Y_v=j)
    co_matrix = torch.matmul(predictions[src].t(), predictions[dst])
    
    # Symmetrize the matrix
    co_matrix = (co_matrix + co_matrix.t()) / 2.0
    
    # Row normalize to get probabilities
    row_sum = co_matrix.sum(dim=1, keepdim=True)
    # Avoid division by zero
    row_sum = torch.clamp(row_sum, min=1e-8)
    co_matrix = co_matrix / row_sum
    
    return co_matrix

def transform_cooccurrence_matrix(co_matrix, mode, generator=None):
    """
    Applies an ablation transform to the co-occurrence matrix before the
    penalty is built.

    Modes:
        'none'      : identity.
        'shuffle'   : randomly permute all entries, then re-row-normalize.
                      Destroys ALL class semantics, including homophily
                      (the diagonal). A model gaining from a shuffled
                      penalty is gaining from generic regularization, not
                      from co-occurrence information.
        'homophily' : keep the diagonal (per-class self-affinity), spread
                      each row's remaining mass uniformly off-diagonal.
                      Preserves the homophily level but removes all
                      class-pair structure. Isolates the contribution of
                      the off-diagonal statistics.

    Note: a simultaneous row/column permutation (class relabeling) is NOT
    used as a placebo — it maps diagonal to diagonal and therefore
    preserves homophily structure, which is the dominant signal on
    homophilic graphs.
    """
    if mode == 'none':
        return co_matrix
    k = co_matrix.shape[0]
    if mode == 'shuffle':
        flat = co_matrix.flatten()
        perm = torch.randperm(flat.numel(), generator=generator).to(co_matrix.device)
        shuffled = flat[perm].reshape(k, k)
        row_sum = shuffled.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return shuffled / row_sum
    if mode == 'homophily':
        if k <= 2:
            print("WARNING: 'homophily' transform is the IDENTITY for k<=2 "
                  "(one off-diagonal entry per row reconstructs exactly) — "
                  "this condition is vacuous on binary datasets and its "
                  "penalty_dist diagnostic will be 0.")
        row_sum = co_matrix.sum(dim=1)
        diag = co_matrix.diagonal()
        if k > 1:
            off = (row_sum - diag) / (k - 1)
        else:
            off = torch.zeros_like(diag)
        out = off.unsqueeze(1).expand(k, k).clone()
        idx = torch.arange(k, device=co_matrix.device)
        out[idx, idx] = diag
        return out
    raise ValueError(f"unknown penalty transform: {mode}")

def penalty_stats(penalty_orig, penalty_new):
    """
    Distinguishability diagnostics for the placebo test.

    Returns:
        rel_dist   : ||P_new - P_orig||_F / ||P_orig||_F. If this is near
                     zero, the transform barely changed the penalty and the
                     placebo comparison has no statistical power on this
                     dataset (e.g., near-uniform co-occurrence statistics).
        offdiag_cv : coefficient of variation of the off-diagonal entries
                     of the ORIGINAL penalty — how much class-pair
                     structure exists to destroy in the first place.
    """
    rel_dist = ((penalty_new - penalty_orig).norm() /
                penalty_orig.norm().clamp(min=1e-12)).item()
    k = penalty_orig.shape[0]
    if k > 1:
        mask = ~torch.eye(k, dtype=torch.bool, device=penalty_orig.device)
        off = penalty_orig[mask]
        offdiag_cv = (off.std() / off.mean().abs().clamp(min=1e-12)).item()
    else:
        offdiag_cv = 0.0
    return rel_dist, offdiag_cv

def edge_loss(node_probs, edge_index, penalty_matrix):
    """
    Computes the regularization loss based on edge endpoints and a penalty matrix.
    
    Args:
        node_probs (Tensor): The predicted probabilities (N x C) for the current batch/graph.
                             Requires gradients.
        edge_index (Tensor): The graph edge indices (2 x E).
        penalty_matrix (Tensor): The (C x C) matrix containing the penalties (-log probabilities).
                                 Does not require gradients.
                                 
    Returns:
        Tensor: A scalar loss value.
    """
    src, dst = edge_index
    
    p_src = node_probs[src]
    p_dst = node_probs[dst]
    
    # Compute: sum_e p_src[e]^T * penalty_matrix * p_dst[e]
    # (E, C) @ (C, C) -> (E, C)
    projected_dst = torch.matmul(p_dst, penalty_matrix)
    
    # (E, C) * (E, C) -> (E, C). Sum along dim=1 -> (E,)
    edge_penalties = (p_src * projected_dst).sum(dim=1)
    
    # The total loss is the mean over all edges
    return edge_penalties.mean()
