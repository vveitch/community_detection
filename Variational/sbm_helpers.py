import tensorflow as tf
import numpy as np

"""
Tensorflow compatible versions of earlier numpy helper functions for computing 'sufficient stats' for graph models
"""


def comp_edge_cts2(A, z, n_comm):
    """
    Computes the number of edges between the n_comm communities in (multi-)graph with adjacency matrix A
    and community memberships comm_idxs.
    Used for inference calculations in SBM-type models
    :param A: 2-D tensor, nxn matrix, adjacency matrix of (multi-)graph
    :param z: 1-D tensor, each entry is an community label of corresponding vertex
    :param n_comm: integer, number of communities
    :return: n_comm x n_comm matrix, edge_counts[k,l] is number of edges between communities k and l
    """

    comm_idxs = []
    for k in range(n_comm):
        comm_idxs.append([i for i, zi in enumerate(z) if zi == k])  # vertices in community k

    edge_cts = np.zeros([n_comm, n_comm])
    for k in range(n_comm):
        for l in range(k, n_comm):
            for i in comm_idxs[k]:
                edge_cts[k, l] += np.sum(A[i, comm_idxs[l]])  # number of edges from vertex i to community l
            if k == l:
                edge_cts[k, l] = edge_cts[k, l] / 2
            edge_cts[l, k] = edge_cts[k, l]

    return edge_cts.astype(np.float32) # for compatability w tf ops


def comp_nn(z, n_comm):
    """
    :param z: 1-D tensor, each entry is an community label of corresponding vertex
    :param n_comm: integer, number of communities
    :return: 2-D tensor, n_comm x n_comm where ret[k,l] is n[l]*n[k] (divided by 2 on diagonal)
    """

    nn = np.zeros([n_comm,n_comm])
    for k in range(n_comm):
        nk = np.sum(z==k)
        for l in range(k,n_comm):
            nl = np.sum(z==l)

            nn[k,l] = nk*nl
            if k == l:
                nn[k,l] = nn[k,l]/2
            nn[l,k] = nn[k,l]

    comm_idxs = []
    for k in range(n_comm):
        comm_idxs.append([i for i, zi in enumerate(z) if zi == k])  # vertices in community k

    return nn.astype(np.float32) # for compatability w tf ops


# # # debugging code
# # or mb
# AA = tf.tile(tf.reshape(tf.multinomial(tf.log([[10., 10.]]), 10),[-1,1]),[1,10])
# zz = tf.multinomial(tf.log([[10., 10.]]), 10)[0]
# nn = tf.constant(3,dtype=tf.int64)
# mytest = tf.py_func(comp_edge_cts2,[AA,zz,nn],tf.float32)