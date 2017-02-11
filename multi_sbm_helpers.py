import numpy as np
from scipy.misc import logsumexp


def comp_edge_cts(A, comm_idxs):
    """
    Computes the number of edges between the n_comm communities in (multi-)graph with adjacency matrix A
    and community memberships comm_idxs.
    Used for inference calculations in SBM-type models
    :param A: nxn matrix, adjacency matrix of (multi-)graph
    :param comm_idxs: length n_comm list of lists, comm_idxs[k] is list of vertices in community k
    :return: n_comm x n_comm matrix, edge_counts[k,l] is number of edges between communities k and l
    """
    n_comm = len(comm_idxs)
    edge_cts = np.zeros([n_comm, n_comm])
    for k in range(n_comm):
        for l in range(k, n_comm):
            for i in comm_idxs[k]:
                edge_cts[k, l] += np.sum(A[i, comm_idxs[l]])  # number of edges from vertex i to community l
            if k == l:
                edge_cts[k, l] = edge_cts[k, l] / 2
            edge_cts[l, k] = edge_cts[k, l]

    return edge_cts


def comp_tot_cts(comm_cts):
    """
    Computes the maximum number of possible edges between the n_comm communities in a simple graph with community
    occupations n.
    Used for inference calculations in SBM-type models

    :param comm_cts: length n_comm list of integers, n_comm[k] is number of vertices in community k
    :return: n_comm x n_comm matrix, tot_cts[k,l] is number of edges between communities k and l
    """
    n_comm = len(comm_cts)
    tot_cts = np.zeros([n_comm, n_comm])
    for k in range(n_comm):
        for l in range(k, n_comm):
            if (k != l):
                tot_cts[k, l] = comm_cts[k] * comm_cts[l]
            else:
                tot_cts[k, l] = comm_cts[k] * (comm_cts[k] - 1) / 2

            tot_cts[l, k] = tot_cts[k, l]

    return tot_cts


def softmax(log_prob):
    """
    Numerically stable (very, very close) approximation to:
    return np.exp(log_prob)/sum(np.exp(log_prob))
    :param log_prob: logs of (unnormalized) probability distribution
    :return: vector of (non-negative) reals that sums to 1
    """
    prob = np.zeros(len(log_prob))
    rescale = log_prob - np.max(log_prob)
    # entries that give "non-negligible" probability (0 to less than ~43 decimal places)
    non_neg = rescale > -100

    # numerically stable version of np.exp(~)/np.sum(np.exp(~))
    # prob[non_neg] = np.exp(log_prob[non_neg] - logsumexp(log_prob[non_neg]))

    prob[non_neg] = np.exp(rescale[non_neg])
    prob[non_neg] = prob[non_neg]/np.sum(prob[non_neg])

    return prob
