"""
Bayesian Degree Corrected Stochastic Block Model
roughly based on Infinite-degree-corrected stochastic block model by Herlau et. al.,
but with a fixed number of cluster sizes and a different update equation for the collapsed Gibbs sampler;
see accompanying documentation
"""

import numpy as np
import sys

sys.path.append("/home/victor/Documents/community_detection/MCMC")
from cgs_llhds import diri_multi_llhd
from multi_sbm_helpers import comp_edge_cts, softmax
from dcsbm_helpers import GD, BD, samp_shape_post_step, samp_rate_post_step, samp_gam_post_step

class gen_data:
    def __init__(self, n, phis, eta):
        """
        :param n: number of vertices in each community
        :param phis: list of probability distributions, phis[l] should be length n[l] and sum to 1
        :param eta: symmetric matrix, eta[k,l] is expected number of edges between vertex in k and vertex in l
        """

        self.n = n
        self.n_vert = sum(n)
        self.n_comm = len(n)
        self.phis = phis
        self.eta = eta

        z = np.repeat(0, self.n_vert)
        acc = 0
        for l in range(self.n_comm - 1):
            acc += self.n[l]
            z[acc: acc + self.n[l + 1]] = l + 1

        self.z = z

        phi = np.repeat(0., self.n_vert)
        phi[0:self.n[0]] = phis[0]
        acc = 0
        for l in range(self.n_comm - 1):
            acc += self.n[l]
            phi[acc: acc + self.n[l + 1]] = phis[l + 1]

        self.phi = phi

        self.A = self.sampleA()

    def sampleA(self):
        """
        Sample an adjacency matrix conditional on all other parameters
        :return ndarray, float. Sampled adjacency matrix
        """
        A = np.zeros([self.n_vert, self.n_vert])

        for i in range(self.n_vert):
            for j in range(i + 1, self.n_vert):
                thetai = self.n[self.z[i]] * self.phi[i]
                thetaj = self.n[self.z[j]] * self.phi[j]
                A[i, j] = np.random.poisson(thetai * thetaj * self.eta[self.z[i], self.z[j]])
                A[j, i] = A[i, j]

        return A


class gen_data_hypers(gen_data):
    "Sample a graph from the Bayesian DCSBM"

    def __init__(self, n_vert, n_comm, alpha, kap, lam, gam):
        """
        :param n_vert: number of vertices in the graph
        :param n_comm: number of communities
        :param alpha: dirichlet prior parameter for community memberships
        :param kap: scalar, gamma dist param
        :param lam: scalar, gamma dist param
        :param gam: scalar, param for deg correction dirichlet dist. Basic block model recovered in gamma->inf limit
        """

        self.n_vert = n_vert
        self.n_comm = n_comm
        self.alpha = alpha
        self.kap = kap
        self.lam = lam
        self.gam = gam

        self.n = np.random.multinomial(n_vert, np.random.dirichlet(np.repeat(alpha, n_comm)))

        z = np.repeat(0,n_vert)
        acc = 0
        for l in range(n_comm-1):
            acc += self.n[l]
            z[acc : acc + self.n[l + 1]] = l+1

        self.z=z

        phi_ls = [np.random.dirichlet(np.repeat(gam, nl)) for nl in self.n]
        phi = np.repeat(0.,self.n_vert)
        phi[0:self.n[0]] = phi_ls[0]
        acc = 0
        for l in range(n_comm-1):
            acc += self.n[l]
            phi[acc : acc + self.n[l + 1]] = phi_ls[l+1]

        self.phis = phi_ls
        self.phi = phi

        eta = np.random.gamma(kap,1./lam,[n_comm,n_comm])
        for k in range(n_comm):
            for l in range(k+1,n_comm):
                eta[k,l] = eta[l,k]
        self.eta = eta

        self.A = self.sampleA()


class cgs:
    """
    Collapsed Gibbs sampler for the Bayesian DCSBM
    """

    def __init__(self, A, z, n_comm, alpha, kap, lam, gam):
        """
        :param A: adjacency matrix of (multi) graph
        :param z: community identities of vertices
        :param n_comm: number of communities (in case some communities are empty at init)
        :param alpha: dirichlet prior parameter for community memberships
        :param kap: scalar, gamma dist param
        :param lam: scalar, gamma dist param
        :param gam: scalar, param for deg correction dirichlet dist. Basic block model recovered in gamma->inf limit
        """

        self.A = A
        self.z = z
        self.n_comm = n_comm
        self.alpha = 1.*alpha
        self.kap = 1.*kap
        self.lam = 1.*lam
        self.gam = 1.*gam

        '''
        compute initial values of sufficient stats
        '''
        # comm_idxs[k] is list of indices of vertices in community k
        self.comm_idxs = []
        for k in range(n_comm):
            self.comm_idxs.append([i for i, zi in enumerate(z) if zi == k])  # vertices in community k

        # n[k] is number of vertices in community k
        self.n = np.array([members.__len__() for members in self.comm_idxs])

        # edge_cts[k,l] is number of edges between community k and community l, not counting self edges
        # warning: not counting self edges!
        self.edge_cts = comp_edge_cts(self.A, self.comm_idxs)

        self.n_vert = A.shape[0]

        # initial assignment for (latent) self-edges of the graph - shouldn't matter too much
        self.diags = np.repeat(0,self.n_vert)

        self.degs = np.sum(A,axis=1) # degree of each vertex

        # [vectors of the form [1 0 0 0], [0 1 0 0], etc., used to call diri-multi-llhd
        self._comm_indicators = [np.identity(n_comm, int)[j,:] for j in range(n_comm)]

    def sample_diags(self):
        """
        Sample the self edges of the graph, which are treated as latent variables.
        Scheme for doing this is to sample eta, phi | z,A and then A_ii | eta,phi,z
        The relevant posterior distributions are derived in the Bayesian DCSBM paper
        """

        eta_diags = np.zeros(self.n_vert ) # eta_diags[v] = eta_z[v]z[v]
        theta = np.zeros(self.n_vert ) # theta_l = n_l * phi_l

        for l in range(self.n_comm):
            #eta
            kap_post = self.kap + self.edge_cts[l,l] + np.sum(self.diags[self.comm_idxs[l]])
            # kap_post = self.kap + self.edge_cts[l,l]
            lam_post = self.lam + self.n[l]**2 /2.

            # numpy uses a different convention for the parameterization of the gamma distribution
            eta_diags[self.comm_idxs[l]] = np.random.gamma(kap_post, 1./lam_post)

            # phi
            gam_post = self.gam + np.sum(self.A[self.comm_idxs[l],:],axis=1) + 2*self.diags[self.comm_idxs[l]]
            theta[self.comm_idxs[l]] = self.n[l] * np.random.dirichlet(gam_post)

        self.diags = np.random.poisson(0.5 * theta**2 * eta_diags)

    """
    Functions to update zs
    """

    def remove_vertex(self,v):
        """
        removes vertex v from the sufficient stats
        :param v: integer, index of vertex to be removed
        """
        edges_to_comm = [np.sum(self.A[v, comm]) for comm in self.comm_idxs] # vv: double checked this, it's fine
        self.comm_idxs[self.z[v]].remove(v)

        self.n[self.z[v]] -= 1

        self.edge_cts[self.z[v], :] -= edges_to_comm
        self.edge_cts[:, self.z[v]] = self.edge_cts[self.z[v], :]

        # set label to invalid value. Should have no effect (except to throw error if invoked before being reassinged)
        self.z[v] = self.n_vert+1

    def add_vertex(self,v,k):
        """
        adds vertex v to community k, updating the sufficient stats of the SBM
        :param v: integer, index of vertex to be added
        :param k: integer, index of community to add vertex to
        """

        # warning! this doesn't include self edges
        edges_to_comm = [np.sum(self.A[v, comm]) for comm in self.comm_idxs]

        self.z[v] = k

        self.comm_idxs[self.z[v]].append(v)
        self.n[k] += 1

        self.edge_cts[self.z[v], :] += edges_to_comm
        self.edge_cts[:, self.z[v]] = self.edge_cts[self.z[v], :]

    def comm_llhd(self, v):
        """
        computes a length n vector q such that q(k)-q(l) = log(P(A|z_-v,z_v =k)) - log(P(A|z_-v,z_v = l))
        See associated documentation for derivation of these equations
        :param v: id of vertex with community identities to be computed
        :return: length n vector q such that q(k)-q(l) = log(P(A|z_-v,z_v =k)) - log(P(A|z_-v,z_v = l))
        """

        # dv_cts[m] is number of edges from v to community m, ignoring the self edges of (the communityless) v
        dv_cts = np.array([np.sum(self.A[v,comm]) for comm in self.comm_idxs])

        tot_trials = 1.*np.outer(self.n, self.n) # number of pairs of vertices between comm k and l
        tot_trials[np.diag_indices_from(tot_trials)] = self.n ** 2 / 2.  # possible connections to comm [l,l] is smaller

        log_comm_prob = np.zeros(self.n_comm)
        for k in range(self.n_comm):
            # SBM component
            edge_add = np.zeros(self.n_comm)
            tot_ct_add = np.zeros(self.n_comm)

            edge_add += dv_cts
            edge_add[k] += self.diags[v]

            tot_ct_add += self.n
            tot_ct_add[k] += 0.5

            log_comm_prob[k] += np.sum(GD(self.edge_cts[k,:]+self.kap, tot_trials[k,:]+self.lam,edge_add,tot_ct_add))

            #degree correction
            if self.n[k]!=0:
                deg_term = np.repeat(self.gam, self.n[k])+self.degs[self.comm_idxs[k]]
                log_comm_prob[k] += BD(deg_term, self.degs[v]+self.diags[v]+self.gam) \
                                    - BD(np.repeat(self.gam,self.n[k]),self.gam)
                log_comm_prob[k] += (np.sum(self.edge_cts[k,:])+self.edge_cts[k,k]+dv_cts[k])*np.log(1.+1./self.n[k]) \
                                    + (self.degs[v]+self.diags[v])*np.log(self.n[k]+1)

        return log_comm_prob

    def log_comm_prior(self):
        """
        :return: log dirichlent multinomial values (prior for z), for pass to softmax
        """
        return np.array([diri_multi_llhd(obs=comm_indic, alphas=self.alpha+self.n) for comm_indic in self._comm_indicators])

    def update_z(self, v):
        """
        Runs a single step of the collapsed gibbs sampler to resample the community identity of v
        :param v: integer, a vertex in the graph
        """

        # sample new self edges of adjacency matrix
        # it's really irritating that this is required
        self.sample_diags()

        # add in the contribution from the self edges to the sufficient stats
        # self_edges[l] is number of self edges in community l
        self_edges = np.array([np.sum(self.diags[comm_idx]) for comm_idx in self.comm_idxs])
        self.edge_cts[np.diag_indices_from(self.edge_cts)] += self_edges
        self.degs += self.diags


        '''
        remove current vertex from sufficient stats
        '''
        self.remove_vertex(v)

        '''
        sample the new index conditional on all other community assignments
        using P(z_v = k | A, z_-v) \propto P( A | z_v = k, z_-v) * P(z_v = k | z_-v)
        '''

        # P( A | z_v = k, z_-v) contribution
        log_comm_prob = self.comm_llhd(v)

        # P(z_v = k | z_-v) contribution
        log_comm_prob += self.log_comm_prior()

        # exponentiate and sample the new label
        comm_prob = softmax(log_comm_prob)
        new_comm = np.random.multinomial(1, comm_prob).nonzero()[0][0]

        '''
        add the vertex back into the sufficient stats
        '''
        self.add_vertex(v,new_comm)

        # clean up the self edges
        self.edge_cts[np.diag_indices_from(self.edge_cts)] -= self_edges
        self.degs -= self.diags

    def update_zs(self):
        """
        Update the community indicators in all of the models
        """
        vertex_order = range(self.n_vert)
        np.random.shuffle(vertex_order)
        for v in vertex_order:
            self.update_z(v)

    """
    Functions to update higher level parameters (kap,lam,gam)
    """

    def update_NB_params(self):
        kap = np.copy(self.kap)
        lam = np.copy(self.lam)

        tot_trials = 1. * np.outer(self.n, self.n)  # number of pairs of vertices between comm k and l
        tot_trials[np.diag_indices_from(tot_trials)] = self.n ** 2 / 2.  # possible connections to comm [l,l] is smaller

        # empty communities carry no information (but do cause divide by 0 errors -_-)
        ttm = np.ma.masked_values(tot_trials,0)
        em = np.ma.masked_array(self.edge_cts, ttm.mask)

        # count each community pair only once
        unique_pairs = np.triu_indices(self.n_comm)


        # update kappa
        # key observation is that e_lm ~ NB(kap, 1/(1+lam/n_lm))
        # so we can use augmented conjugate update of Zhou&Carin 2012
        ps = ttm[unique_pairs] / (ttm[unique_pairs] + lam)
        kap = samp_shape_post_step(em[unique_pairs].compressed(), kap, ps.compressed(), 0.1, 0.1)

        # update lambda
        # simple independent MH sampler
        lam = samp_rate_post_step(em[unique_pairs].compressed(), ttm[unique_pairs].compressed(), kap, lam)
        self.kap = kap
        self.lam = lam

    def update_gam(self):
        # metropolis-hasting update of gamma
        gam = np.copy(self.gam)

        terms = self.degs + 2*self.diags
        gam = samp_gam_post_step(terms, self.comm_idxs, gam)

        self.gam = gam
