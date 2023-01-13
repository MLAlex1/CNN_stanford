import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    norm_dot_product = (z_i @ z_j) / (torch.linalg.norm(z_i) *  torch.linalg.norm(z_j))
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        # z_k and z_k_N are positive pairs, while z_k with all other z != z_k_N are negative
        # pairs and similarly z_k_N with all other z != z_k are negative pairs!
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # skip i != k and i != k + N, same as indicator 1
        denom_l = torch.exp(torch.tensor([sim(z_k, out[i])/tau for i in range(2*N) if i != k])).sum()
        denom_r = torch.exp(torch.tensor([sim(z_k_N, out[i])/tau for i in range(2*N) if i != k + N])).sum()
        # num the same as sim(z_k, z_k_N) = sim(z_k_N, z_k)
        num = torch.exp(sim(z_k, z_k_N)/tau)
        l_left  = -torch.log(num/denom_l)
        l_right = -torch.log(num/denom_r)
        total_loss += l_left + l_right
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # normalised left
    norm_left = out_left/torch.linalg.norm(out_left, dim=1).view(-1,1)
    # normalised right
    norm_right = out_right/torch.linalg.norm(out_right, dim=1).view(-1,1)
    # transpose norm_right so we can perform matrix mult. We then have a matrix
    # with similarities for each vector, but we only need diag elements as we
    # only extract positive similarities!
    # pos_pairs = (norm_left @ (norm_right).T).diag()
    # better implementation as we don't need to compute entire matrix of similarities:
    # we simply moltiply and sum across dim=1 which is basically the dot product
    pos_pairs = (norm_left * norm_right).sum(dim=1)
    
    # reshape to output Nx1 tensor
    pos_pairs = pos_pairs.view(-1, 1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    out_norm = out/torch.linalg.norm(out, dim=1).view(-1,1)
    sim_matrix = out_norm @ out_norm.T
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cpu'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    exponential = torch.exp(sim_matrix/tau)
    
    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = exponential.sum(dim=1, keepdim=True)

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    pos_pair_sim = sim_matrix[torch.arange(0, N), torch.arange(N, 2*N)][:, None]
    
    # Step 3: Compute the numerator value for all augmented samples.
    numerator = (pos_pair_sim / tau).exp().repeat(2, 1)

    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = -(numerator / denom).log().mean()
    return loss

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))