import numpy as np
import matplotlib.pyplot as plt

import dcsbm

def main():
    """
    Example of use of collapsed gibbs sampler for degree corrected stochastic block model
    """

    # np.random.seed(1)  # reproducibility
    np.seterr(all='raise')

    '''
    data
    '''
    # simulate some data to test
    # ground truth is z=[0 ... 0 1 .. 1 2 .. 2] with community sizes Diri-Multi(np.repeat(alpha,n_comm))
    # kap/lam is expected number of edges between any pair of vertices
    # kap/lam^2 determines how well separated the community weights are (variance of the gamma)
    data = dcsbm.gen_data_hypers(n_vert=100, n_comm=3, alpha=10, kap=2.5, lam=5, gam=5)

    # heatmap of graph
    A = data.A
    plt.imshow(A, origin='upper', interpolation='nearest')
    plt.show()

    '''
    set up the model
    '''
    # model parameters
    n_iter = 100 # full iterations of gibbs sampler
    n_comm = 4 # number of communities in the graph
    alpha = 5. # parameter for dirichlet prior on communities
    kap = 10. # parameter for gamma prior on edge weights
    lam = 0.01 # parameter for gamma prior on edge weights
    gam = 1. # parameter for dirichlet prior for degree correction; classic sbm in the limit gamma -> infty

    n_vert = A.shape[0]

    # initial community assignments
    z_init = np.random.randint(low=0, high=n_comm, size=n_vert)

    # initialize the collapsed gibbs sampler
    model = dcsbm.cgs(A=A, z=z_init, n_comm=n_comm, alpha=alpha, kap=kap, lam=lam, gam=gam)

    # fit the model
    for itr in range(n_iter):
        if np.mod(itr,10) == 0:
            print 'z:', model.z
            print 'lam:', model.lam
            print 'kap', model.kap
            print 'gam', model.gam
            print 'ratio', model.kap / model.lam

        # model.update_NB_params()
        model.update_zs()
        model.update_NB_params()
        model.update_gam()

if __name__ == '__main__':
    main()
