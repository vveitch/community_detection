import numpy as np
import tensorflow as tf
import edward as ed

from edward.models import RandomVariable
from tensorflow.contrib.distributions import Distribution

from sbm_helpers import comp_edge_cts2, comp_nn

class SBM(RandomVariable, Distribution):
    def __init__(self, zs, eta, n_comm, *args, **kwargs):
        """

        :param zs:
        :param eta:
        :param n_comm:
        :param args:
        :param kwargs:
        """
        super(SBM, self).__init__(*args, **kwargs)

    def _log_prob(self, A):
        """

        :param A: A 2-D 'tensor' representing adjacency matrix of graph; float32 for ease of tf
        :return: _log_prob(X)
        """

        # number of edges between each pair of communities
        ec = tf.py_func(comp_edge_cts2, [A, self.zs, self.n_comm], [tf.float32])
        nn = tf.py_func(comp_nn, [self.zs, self.n_comm], [tf.float32])

        # unnormalized
        lp = tf.reduce_sum(tf.matrix_band_part(ec*tf.log(self.eta) - self.eta * nn),0,-1)

        # normalize
        uppA = tf.matrix_band_part(A, 0, -1)
        return lp - tf.reduce_sum(tf.diagpart(A)*tf.log(2.) + tf.lgamma(uppA+1.))

    def _sample_n(self, n, seed=None):
        # TBD: I'm not sure about the 'batch_shape' stuff... this may not be idiomatic w tf samplers

        def np_one_samp(zs,eta):
            n_vert = len(zs)
            A = np.zeros([n_vert,n_vert])
            for i in range(n_vert):
                for j in range(i,n_vert):
                    if 1 != j:
                        A[i,j] = np.random.poisson(eta[zs[i],zs[j]])
                        A[j,i] = A[i,j]
                    else:
                        A[i,i] = np.random.poisson(eta[zs[i],zs[j]] / 2.)

            return A.astype(float)

        def np_samp(n,zs,eta):
            return np.array([np_one_samp(zs,eta) for _ in range(n)])

        tf.py_func(np_samp, [n, self.zs, self.eta], tf.float32)
