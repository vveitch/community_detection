import numpy as np

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

import edward as ed

from edward.models import RandomVariable
from tensorflow.contrib.distributions import Distribution
from tensorflow.contrib.distributions.python.ops import distribution

from sbm_helpers import comp_edge_cts2, comp_nn

# class SBM(RandomVariable, Distribution):
class SBM(Distribution):
    def __init__(self,
                 zs,
                 eta,
                 n_comm,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="SBM"):
                 # *args, **kwargs):
        """

        :param zs:
        :param eta:
        :param n_comm:
        :param dtype:
        :param validate_args:
        :param allow_nan_stats:
        :param name:
        :param args:
        :param kwargs:
        """

        parameters = locals()

        with ops.name_scope(name, values=[zs, eta, n_comm]):
            self._zs = ops.convert_to_tensor(zs, name="zs")
            self._eta = ops.convert_to_tensor(eta, name="eta")
            self._n_comm = ops.convert_to_tensor(zs, name="n_comm")

        # super(SBM, self).__init__(
        #     dtype=dtypes.float32,
        #     reparameterization_type=distribution.NOT_REPARAMETERIZED,
        #     validate_args=validate_args,
        #     allow_nan_stats=allow_nan_stats,
        #     parameters=parameters,
        #     graph_parents=[self._zs, self._eta, self._n_comm],
        #     name=name)
        #     # *args, **kwargs)

    def _sample_n(self, n=1, seed=None):
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

            return A.astype(np.float32)

        def np_samp(n_samp, zs, eta):
            return np.array([np_one_samp(zs,eta) for _ in range(n_samp)]).astype(np.float32)

        return tf.py_func(np_samp, [n, self._zs, self._eta], [tf.float32])[0]


    def _log_prob(self, A):
        """

        :param A: A 2-D 'tensor' representing adjacency matrix of graph; float32 for ease of tf
        :return: _log_prob(X)
        """

        zs = self._zs
        eta = self._eta
        n_comm = self._n_comm

        # number of edges between each pair of communities
        ec = tf.py_func(comp_edge_cts2, [A, zs, n_comm], [tf.float32])[0]
        nn = tf.py_func(comp_nn, [zs, n_comm], [tf.float32])[0]

        # unnormalized
        lp = tf.reduce_sum(tf.matrix_band_part(ec * tf.log(eta) - eta * nn, 0, -1))

        # normalize
        uppA = tf.matrix_band_part(A, 0, -1)
        return lp - tf.reduce_sum(tf.diagpart(A) * tf.log(2.) + tf.lgamma(uppA + 1.))
