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
