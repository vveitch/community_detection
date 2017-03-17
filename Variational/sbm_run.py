import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Dirichlet, Categorical, Gamma, Poisson

# debugging
import sys
sys.path.append("/home/victor/Documents/community_detection/Variational")

from sbm import SBM

# SBM parameters
n_vert = 10
n_comm = 3
n_comm_t = tf.constant(n_comm)

alpha = tf.Variable(3.0,dtype=tf.float32)
lam = tf.Variable(1,dtype=tf.float32)
kap = tf.Variable(1,dtype=tf.float32)

# Model
pi = Dirichlet(alpha=tf.ones([n_comm]))
z = Categorical(p=tf.ones([n_vert, n_comm]) * pi) # z.sample().eval()

eta = Gamma(tf.ones([n_comm, n_comm]), tf.ones([n_comm, n_comm]))

g = SBM(z,eta,3)
x = g._sample_n(1)

sess = tf.InteractiveSession()
print x.eval()

# Variational posterior
# qpi = Dirichlet( alpha = tf.Variable(tf.ones([n_comm])) )
# qz = Categorical( p = tf.Variable(tf.ones([n_comm])/n_comm))
# qeta = Gamma(tf.Variable(tf.ones([n_comm, n_comm])), tf.Variable(tf.ones([n_comm, n_comm])))

# Inference
# TBD... presumably variational EM