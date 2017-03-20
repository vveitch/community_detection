import edward as ed
import tensorflow as tf

from edward.models import Dirichlet, Categorical, Gamma

# debugging
import sys
sys.path.append("/home/victor/Documents/community_detection/Variational")

from sbm import SBM

# SBM parameters
n_vert = 100
n_comm = 3

# fake a dataset
# sort the ground truth community identities to make it easy to parse them
z_gt = tf.Variable(tf.nn.top_k(Categorical(p=tf.ones([n_vert, n_comm])/n_comm).sample(),k=n_vert).values)
eta_gt = tf.Variable(Gamma(tf.ones([n_comm, n_comm]), tf.ones([n_comm, n_comm])).sample())
g=SBM(zs = z_gt, eta = eta_gt, n_comm = n_comm)
data = SBM(zs = z_gt, eta = eta_gt, n_comm = n_comm).sample()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    dataset = data.eval()
    z_gt = z_gt.eval()
    eta_gt = eta_gt.eval()

# Model

# higher level parameters
# alpha = tf.Variable(3.0,dtype=tf.float32)
# lam = tf.Variable(1,dtype=tf.float32)
# kap = tf.Variable(1,dtype=tf.float32)

# communities
# pi = Dirichlet(alpha=alpha*tf.ones([n_comm]))
# z = Categorical(p=tf.ones([n_vert, n_comm]) * pi)
z = Categorical(p=tf.ones([n_vert, n_comm]) / 3. ) # z.sample().eval()

# comm-comm weights
# eta = Gamma(lam*tf.ones([n_comm, n_comm]), kap*tf.ones([n_comm, n_comm]))
eta = Gamma(tf.ones([n_comm, n_comm]), tf.ones([n_comm, n_comm]))

g = SBM(zs=z,eta=eta,n_comm=n_comm)

# Variational posterior
# qpi = Dirichlet( alpha = tf.Variable(tf.ones([n_comm])) )
qz = Categorical( tf.Variable(tf.ones([n_vert, n_comm])))
qeta = Gamma(tf.Variable(tf.ones([n_comm, n_comm])), tf.Variable(tf.ones([n_comm, n_comm])))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    print g.log_prob(dataset).eval()

# Inference
inference = ed.KLqp({z: qz}, data={g: dataset, eta: eta_gt})


inference.initialize(n_samples=20, n_iter=10000)

tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)


# sess = tf.InteractiveSession()
# init = tf.global_variables_initializer()
# init.run()
# dataset.eval()