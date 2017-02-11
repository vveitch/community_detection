import numpy as np

import dcsbm
import multi_dcsbm as mdcsbm

from multi_sbm_helpers import comp_edge_cts

np.seterr(all='raise')

def sim_data(comm_sizes=[30, 40, 40],gam=100, n_graphs=100,n_types=2):
    """
    Simulate a dataset from the multi-sbm model
    :return As, a list of 100 adjacency matrices of graphs on 100 vertices,
    gen according to multisbm model with parameters specified in code below, and
    covariates, a list of covariates (categorical, one for each graph)
    """

    n_comm = np.alen(comm_sizes)

    # to make the problem harder, we set this equal for both types
    # so the algorithm has to guess types based only on degree dists, accounting for community structure
    # eta[k,l] is the expected number of edges between pair of vertices in comms k,l
    eta = np.empty([n_types, n_comm, n_comm])
    eta[0, :, :] = np.array([[0.5, 0.3, 0.4], [0.3, 0.5, 0.3], [0.45, 0.3, 0.5]])
    eta[1, :, :] = eta[0, :, :]

    phi_ls = [[np.random.dirichlet(np.repeat(gam, n)) for n in comm_sizes] for _ in range(n_types)]

    true_t = np.append(np.repeat(0, np.floor(0.4 * n_graphs)),
                       np.repeat(1, np.floor(0.6 * n_graphs)))  # identities for graph types

    As = [dcsbm.gen_data(comm_sizes, phi_ls[s], eta[s,:]).A for s in true_t]

    return As

def main():
    """
    Example of use of collapsed gibbs sampler for multi-graph degree corrected stochastic block model
    """

    '''
    data
    '''
    # ground truth by default:
    # same community connection weights (eta) for both types
    # gam = 100 and comm_sizes = [30, 40, 40] for both types
    # t = [0 ... 0 1 .. 1] with 40 in first type and 60 in second type
    As = sim_data()

    '''
    model setup
    '''
    # model parameters
    n_iter = 100 # full iterations of gibbs samper
    n_types = 3  # number of different brain types
    n_comm = np.repeat(4, n_types)  # number of SBM communities in each brain type
    kap = 100.
    lam = 1  # priors for SBM (same for all types)
    gam = 10 # prior for degree correction (same for all types)
    alpha = 1  # dirichlet priors for comm assignment
    beta = 1 # dirichlet prior for type assignment

    n_graphs = len(As)
    n_vert = As[0].shape[0]

    # initial type and community assignments
    ts = np.random.randint(low=0, high=n_types, size=n_graphs)
    zs = np.empty([n_types, n_vert], dtype=int)
    for s in range(n_types):
         zs[s] = np.random.randint(low=0, high=n_comm[s], size=n_vert)

    z0 = np.random.randint(low=0, high=n_comm[0], size=n_vert)

    # Two possible models: either each graph type has its own distinct community assignments,
    # or all community assignments are shared across type
    # model = mdcsbm.cgs(As,n_types,ts,n_comm,zs,alpha,kap,lam,gam,beta)
    model = mdcsbm.cgsSharedComm(As,n_types,ts,n_comm[0],z0,alpha,kap,lam,gam,beta)

    for itr in range(n_iter):
        if np.mod(itr, 5) == 0:
            print 'z:', model.zs
            print "t:", model.ts

        model.update_zs()
        model.update_ts()
        model.update_NB_params_joint()
        # model.update_NB_params_local() # update each of these separately if community ids aren't shared
        model.update_gam_local()

    # print the most recent samples for the higher level parameters
    for m in model.agg_models:
        print "gam", m.gam_base
        print "lam", m.lam_base
        print "kap", m.kap

if __name__ == '__main__':
    main()