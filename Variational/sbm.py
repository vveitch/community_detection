import numpy as np

import tensorflow as tf

from edward.models.random_variable import RandomVariable
from tensorflow.contrib.distributions import Distribution

from sbm_helpers import comp_edge_cts2, comp_nn


class SBM(RandomVariable, Distribution):
    def __init__(self,
                 zs,
                 eta,
                 n_comm,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="SBM",
                 value=None
                 ):
        """
        The (Poisson) stochastic block model.

        :param zs: tf.Tensor, 1-D; zs[v] is comm id of vertex v
        :param eta: tf.Tensor, 2-D; eta[k,l] is expected number of edges between a vertex in comm l and one in comm k
        :param n_comm: scalar, number of communities
        """

        parameters = locals()

        with tf.name_scope(name, values=[zs, eta, n_comm]) as ns:
            # TODO: does this screw up composition?
            with tf.control_dependencies([]):
                self._zs = tf.convert_to_tensor(zs, name="zs", dtype=tf.int32)
                self._eta = tf.convert_to_tensor(eta, name="eta", dtype=tf.float32)
                self._n_comm = tf.convert_to_tensor(n_comm, name="n_comm", dtype=tf.int32)

        super(SBM, self).__init__(
            dtype=tf.float32,
            parameters=parameters,
            is_continuous=False,
            is_reparameterized=False,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=ns,
            value=value)

        # TODO: I put this in to mimic the behaviour of the tf distributions, but I dunno...
        self._kwargs = {"zs": self._zs,
                        "eta": self._eta,
                        "n_comm": self._n_comm}

    @property
    def zs(self):
        """Community ids."""
        return self._zs

    @property
    def eta(self):
        """Link weights"""
        return self._eta

    @property
    def eta(self):
        """number of communities"""
        return self._n_comm

    def _batch_shape(self):
        return tf.convert_to_tensor(self.get_batch_shape())

    def _get_batch_shape(self):
        # TODO: not sure about this
        return self._n_comm.get_shape()

    def _event_shape(self):
        return tf.convert_to_tensor(self.get_event_shape())

    def _get_event_shape(self):
        # TODO: check this works as expected with broadcasting
        n_vert = self._zs.get_shape()[0]
        return tf.TensorShape([n_vert, n_vert])

    def _sample_n(self, n=1, seed=None):
        # TODO: this won't broadcast correctly

        if seed is not None:
            raise NotImplementedError("seed is not implemented.")

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

        :param A: A 2-D 'tensor' representing adjacency matrix of graph; float32
        :return: _log_prob(X)
        """

        At = tf.convert_to_tensor(A)

        zs = self._zs
        eta = self._eta
        n_comm = self._n_comm

        # number of edges between each pair of communities
        ec = tf.py_func(comp_edge_cts2, [At, zs, n_comm], [tf.float32])[0]
        nn = tf.py_func(comp_nn, [zs, n_comm], [tf.float32])[0]

        # unnormalized
        lp = tf.reduce_sum(tf.matrix_band_part(ec * tf.log(eta) - eta * nn, 0, -1))

        # normalize
        uppA = tf.matrix_band_part(At, 0, -1)
        return lp - tf.reduce_sum(tf.diag_part(At) * tf.log(2.) + tf.lgamma(uppA + 1.))
