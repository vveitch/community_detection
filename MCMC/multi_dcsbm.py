import numpy as np
from multi_sbm_helpers import comp_edge_cts, softmax
from cgs_llhds import diri_multi_llhd
from dcsbm_helpers import GD, BD2, samp_shape_post_step
from multi_dcsbm_helpers import mdcsbm_samp_rate_post_step
from itertools import compress

import dcsbm

class aug_dcsbm_cgs(dcsbm.cgs):
    """
    Augmented collapsed gibbs sampler for the DCSBM.
    Supports methods to add and remove graphs.
    Basic idea is that we're dealing with the case where A is the sum of the adjacency matrices of many graphs (so it's
    a simultaneous DCSBM on many graphs), and we need methods to add and remove graphs to the sum of A.
    Used for the multi-DCSBM collapsed gibbs sampler
    """

    def __init__(self, A, z, n_comm, alpha, kap_base, lam_base, gam_base, n_graphs=1):
        """
        :param A: adjacency matrix of (multi) graph (sum of participant graphs)
        :param z: community identities of vertices
        :param n_comm: number of communities (in case some communities are empty at init)
        :param alpha: dirichlet prior parameter for community memberships
        :param kap: scalar, gamma dist param; for single graph
        :param lam: scalar, gamma dist param; for single graph
        :param gam: scalar, param for deg correction dirichlet dist; for single graph.
        Basic block model recovered in gamma->inf limit
        :param n_graphs: int, number of graphs that have been aggregated into A
        """

        # kap_base and gam_base not actually used for anything
        self.kap_base = kap_base
        self.lam_base = lam_base
        self.gam_base = gam_base
        self.n_graphs = n_graphs

        # the DCSBM for A = sum_i A_i
        if n_graphs != 0:
            dcsbm.cgs.__init__(self, A, z, n_comm, alpha, kap_base, 1.*lam_base / n_graphs, gam_base)
        else:
            # this will cause all of the sufficient stats to initialize appropriately, but it's not
            # really quite the right thing... might cause problems if used incautiously
            A = np.zeros([len(z),len(z)])
            dcsbm.cgs.__init__(self, A, z, n_comm, alpha, kap_base, lam_base, gam_base)

    def remove_graph(self, A_cut, A_cut_diags):
        """
        In the case that self.A is actually the sum of many (simple) adjacency matrices, this cuts out one of those
        :param A_cut: nparray. graph to be removed
        :param A_cut_diags: (imputed) diagonal entries of A_cut
        """

        self.n_graphs -= 1  # 1 less possible edge between any pair of vertices

        if self.n_graphs==0:
            self.lam=-1. # this should never be invoked anywhere...
        else:
            self.lam = 1.*self.lam_base / self.n_graphs

        self.A = self.A - A_cut
        self.diags = self.diags - A_cut_diags

        # edge_cts[k,l] is number of edges between community k and community l
        self.edge_cts = comp_edge_cts(self.A,self.comm_idxs)

        self.degs = np.sum(self.A, axis=1)

    def add_graph(self, A_add, A_add_diags):
        """
        In the case that self.A is actually the sum of many (simple) adjacency matrices, this cuts out one of those
        :param A_add: graph to be added
        :param A_add_diags: (imputed) diagonal entries of A_add
        """

        self.n_graphs += 1  # 1 more possible edge between any pair of vertices

        self.lam = 1. * self.lam_base / self.n_graphs

        self.A = self.A + A_add
        self.diags = self.diags + A_add_diags

        # edge_cts[k,l] is number of edges between community k and community l (ignoring self edges)
        self.edge_cts = comp_edge_cts(self.A, self.comm_idxs)

        self.degs = np.sum(self.A, axis=1)

    def q(self, A_new, A_new_diags):
        """
        Computes q(A_new, self.z) where, for A and adjacency matrix of a single (multi)-graph,
        log(P(A|z,lam,gam,kap)) = q(A, z, lam, gam, kap) + C(A) where C(A) depends only on A (and thus carries no info about how compatible
        A is with this graph model)
        More precisely, C(A) = -sum_i\lej gammaln(A_ij) - log(2)*sum_i A_ii
        Expression for P(A|z,lam,gam,kap) from DCSBM paper
        This computation is used in the gibbs update for the type assignments of graphs in the multi-DCSBM model
        :param A_new: nparray, adjacency matrix
        :param A_new_diags, nparray, the self edges of A_new (which ought to be imputed at an earlier stage?)
        :return: npfloat, q(A_new)
        """

        # Compute the sufficient stats, then use these to compute llhd based on expression from DCSBM
        tot_trials = 1. * np.outer(self.n, self.n)  # number of pairs of vertices between comm k and l
        tot_trials[np.diag_indices_from(tot_trials)] = self.n ** 2 / 2.  # possible connections to comm [l,l] is smaller

        if self.n_graphs == 0:
            # if we're considering a type that currently contains no graphs
            kap_post = self.kap_base + np.zeros([self.n_comm,self.n_comm])
            lam_post = self.lam_base + tot_trials
            gam_post = self.gam_base + np.zeros(len(self.z))
        else:
            # # sample new self edges of adjacency matrix for the current multigraph model
            self.sample_diags()

            # add in the contribution from the self edges to the sufficient stats
            # self_edges[l] is number of self edges in community l
            self_edges = np.array([np.sum(self.diags[comm_idx]) for comm_idx in self.comm_idxs])
            self.edge_cts[np.diag_indices_from(self.edge_cts)] += self_edges
            self.degs += self.diags

            kap_post = self.kap + self.edge_cts
            lam_post = self.n_graphs * (self.lam + tot_trials)
            gam_post = self.gam + self.diags + self.degs

        # compute sufficient stats of A_new
        # note that sufficient stats for terms that only depend on comm identity as the same as for aggregate graph

        # for debugging
        # A_new_diags = np.random.binomial(self.diags, 1. / self.n_graphs)

        A_self_edges = np.array([np.sum(A_new_diags[comm_idx]) for comm_idx in self.comm_idxs])
        A_ec = comp_edge_cts(A_new,self.comm_idxs)
        A_ec[np.diag_indices_from(A_ec)] += A_self_edges
        # number of termini incident on each vertex
        A_terms = np.sum(A_new, axis=1) + 2 * A_new_diags

        q = 0

        for k in range(self.n_comm):
            for l in range(k,self.n_comm):
                q += GD(kap_post[k,l], lam_post[k,l], A_ec[k,l], tot_trials[k,l])

        for l in range(self.n_comm):
            comm_idx = self.comm_idxs[l]
            if len(comm_idx)!=0:
                q += BD2(gam_post[comm_idx], A_terms[comm_idx]) \
                        + np.sum(A_terms[comm_idx])*np.log(self.n[l])

        # clean up the self edges
        if self.n_graphs != 0:
            self.edge_cts[np.diag_indices_from(self.edge_cts)] -= self_edges
            self.degs -= self.diags
        return np.float(q)

class cgs(aug_dcsbm_cgs):
    """
    Collapsed Gibbs sampling for the multi-DCSBM model with no covariates
    """

    def __init__(self, As, n_types, ts, n_comms, zs, alpha, kap, lam, gam, beta):
        """
        :param As: list of adjacency matrices of simple graphs on a common vertex set
        :param cs: list of covariate indicators
        :param n_types: int, number of distinct graph model types
        :param ts: [len(As)], type identity of the graphs
        :param n_comms: [n_types], number of communities in each graph type
        :param zs: [n_types, n_verts], z[t,v] is community membership of vertex v in type t
        :param alpha: dirichlet prior parameter for community memberships, common to all types
        :param kap: scalar, gamma dist param, common to all communities and types
        :param lam: scalar, gamma dist param, common to all communities and types
        :param gam: scalar, param for deg correction dirichlet dist,common to all types -- Basic block model recovered in gamma->inf limit
        :param beta: dirichlet prior for type assignments, common for all types
        """

        self.As = As
        self.n_types = n_types
        self.ts = ts
        self.n_comms = n_comms
        self.zs = zs
        self.kap = np.float32(kap)
        self.lam = np.float32(lam)
        self.gam = np.float32(gam)
        self.alpha = np.float32(alpha)
        self.beta = np.float32(beta)

        self.n_graphs = len(As)
        self.n_vert = As[0].shape[0]

        # type_idxs[s] is list of indices of graphs with type s
        self.type_idxs = []
        for s in range(n_types):
            self.type_idxs.append([i for i, t in enumerate(ts) if t == s])

        # type_cts[s] is number of graphs with type s
        self.type_cts = np.array([members.__len__() for members in self.type_idxs])

        # sufficient stats for type assignment is just sums of all adj mats in that type
        A_sums = [ sum([As[s] for s in type_s])
                   for type_s in self.type_idxs]

        # DCSBM collapsed gibb sampler objects to store the models of each type
        self.agg_models = [aug_dcsbm_cgs(A_sums[s], self.zs[s, :], self.n_comms[s], alpha, self.kap, self.lam,
                                         self.gam, self.type_cts[s]) for s in range(self.n_types)]

        # [vectors of the form [1 0 0 0], [0 1 0 0], etc., used to call diri-multi-llhd
        self._type_indicators = [np.identity(n_types, int)[j, :] for j in range(n_types)]

    def update_model_zs(self, s):
        """
        Run the CGS to update the community identities in model s
        :param s: integer, id of model to be updated
        """
        vertex_order = range(self.n_vert)
        np.random.shuffle(vertex_order)
        for v in vertex_order:
            self.agg_models[s].update_z(v)

    def update_zs(self):
        """
        Update the community indicators in all of the models
        """
        for s in range(self.n_types):
            #update the z only if there's at least one graph to get info from
            if self.type_cts[s] != 0:
                self.update_model_zs(s)

    def update_NB_params_local(self):
        # allow diff models to have different kap and lam
        for m in self.agg_models:
            if m.n_graphs != 0:
                m.update_NB_params()
                m.lam_base = m.n_graphs*m.lam

    def update_gam_local(self):
        for m in self.agg_models:
            m.update_gam()

    def type_llhd(self, g, g_self_edges):
        """
        computes a length n_types vector q such that q(k)-q(l) = log(P(A[g]|z, t[g]=k)) - log(P(A[g]|z, t[g]=l))
        :param g: int, id of graph with type identities to be computed
        :param g_self_edges: nparray, imputed self edges of graph g
        :return: nparray, length n_types vector q such that q(k)-q(l) = log(P(A[g]|z, t[g]=k)) - log(P(A[g]|z, t[g]=l))
        """
        ret = np.asarray([mod.q(self.As[g], g_self_edges) for mod in self.agg_models])
        return ret

    def update_ts(self):
        graph_order = range(self.n_graphs)
        np.random.shuffle(graph_order)

        for g in graph_order:

            # impute self edges for g to be used in llhd computations
            g_self_edges = np.random.binomial(self.agg_models[self.ts[g]].diags, 1./self.type_cts[self.ts[g]])

            # remove graph from its current type (affects llhd of graph under model)
            self.agg_models[self.ts[g]].remove_graph(self.As[g], g_self_edges)

            self.type_cts[self.ts[g]] -= 1  # remove graph from type count (affects diri-multi prob)

            # should do nothing... but will cause an exception to be thrown if this is referenced before it's reassigned
            self.ts[g] = self.n_types + 1

            '''
            Sample type of g from distribution given all other type indicators and all community indicators
            Pr(t_g = s | t_\g, z, As[g]) = Pr(t_g = s | t_\g, agg_models, As[g])
                                         \propto Pr(As[g] | agg_models[s]) * Pr(t_g = s | t_\g)
            '''
            # Pr(As[g] | agg_models[s]) term
            log_type_prob = self.type_llhd(g, g_self_edges)
            # log_type_prob = np.repeat(-1.,self.n_types)

            # Pr(t_g = s | t_\g) term
            for s in range(self.n_types):
                log_type_prob[s] = log_type_prob[s] + \
                                   diri_multi_llhd(obs=self._type_indicators[s], alphas=self.beta + self.type_cts)

            # exponentiate and sample
            type_prob = softmax(log_type_prob)
            self.ts[g] = np.random.multinomial(1, type_prob).nonzero()[0][0]

            # add the graph to its new type
            self.agg_models[self.ts[g]].add_graph(self.As[g],g_self_edges)

            self.type_cts[self.ts[g]] += 1  # add graph to type count (affects diri-multi prob)

class cgsSharedComm(cgs):
    """
    Collapsed Gibbs sampling for the multi-DCSBM model where communities are shared across all distinct types
    """

    def __init__(self, As, n_types, ts, n_comm, zs, alpha, kap, lam, gam, beta):

        # make copies of the initial community assignment stuff so I can reuse the cgs init statement
        multi_zs = np.tile(zs,(n_types,1))
        n_comms = np.repeat(n_comm, n_types)

        cgs.__init__(self, As, n_types, ts, n_comms, multi_zs, alpha, kap, lam, gam, beta)

        # because community indicators are common across all graphs in this model
        self.n_comm = n_comm
        self.zs = zs
        self.n = self.agg_models[0].n # valid since all comm identities are common

        # [vectors of the form [1 0 0 0], [0 1 0 0], etc., used to call diri-multi-llhd
        self._comm_indicators = [np.identity(n_comm, int)[j,:] for j in range(n_comm)]

    def update_z(self, v):
        """
        Runs a single step of the collapsed gibbs sampler to resample the community identity of v
        :param v: integer, a vertex in the graph
        """

        '''
        update diagonal estimation and remove current vertex from sufficient stats
        '''
        self_edges = []
        for m in self.agg_models:
            if m.n_graphs != 0:
                # add in the contribution from the self edges to the sufficient stats
                # self_edges[l] is number of self edges in community l
                m.sample_diags()
                m_self_edges = np.array([np.sum(m.diags[comm_idx]) for comm_idx in m.comm_idxs])
                m.edge_cts[np.diag_indices_from(m.edge_cts)] += m_self_edges
                m.degs += m.diags
                self_edges.append(m_self_edges) # remember these to kill em later

            m.remove_vertex(v)

        # valid because community indicators are same across all graphs
        self.n = self.agg_models[0].n

        '''
        sample the new index conditional on all other community assignments and the type assignments
        using P(z_i = k | A, z_\i) \propto \prod_s P( A[s][i,:] \given z_i = k, z_\i, ts) * P(z_i = k | z_\i)
        '''
        log_comm_prob = np.zeros(self.n_comm)
        # TBD: computations below could be vectorized to maybe speed up the code by a factor of 2.

        # this is the bit where the common community id's come into play
        # \prod_s P( A[s][i,:] \given z_i = k, z_\i)
        for m in self.agg_models:
            if m.n_graphs != 0:
                log_comm_prob += m.comm_llhd(v)


        # P(z_i = k | z_\i) (diri-multinom) part of the likelihood
        log_comm_prob = log_comm_prob + \
                        np.array([diri_multi_llhd(obs=comm_indic, alphas=self.alpha + self.n) for comm_indic in self._comm_indicators])

        # exponentiate and sample the new label
        comm_prob = softmax(log_comm_prob)
        new_comm = np.random.multinomial(1, comm_prob).nonzero()[0][0]

        '''
        add the vertex back into the sufficient stats and clean up self edges
        '''
        itr = 0
        for m in self.agg_models:
            m.add_vertex(v,new_comm)

            if m.n_graphs != 0:
                m.degs -= m.diags
                m.edge_cts[np.diag_indices_from(m.edge_cts)] -= self_edges[itr]
                itr+=1

        # valid because community indicators are same across all graphs
        self.n = self.agg_models[0].n

    def update_zs(self):
        """
        Update the community indicators in all of the models
        """
        vertex_order = range(self.n_vert)
        np.random.shuffle(vertex_order)
        for v in vertex_order:
            self.update_z(v)

        self.zs = self.agg_models[0].z


    def update_NB_params_joint(self):
        """
        Update the parameters of the negative binomial distribution (governing edge rates), sharing information between
        distinct graph types
        """
        kap = np.copy(self.kap)
        lam = np.copy(self.lam)

        # the community occupancy counts and edge counts for each community pair
        t_upp = []
        e_upp = []
        n_graphs = []

        relevant = [m.n_graphs != 0 for m in self.agg_models]

        for m in compress(self.agg_models,relevant):
            tot_trials = 1. * np.outer(m.n, m.n)  # number of pairs of vertices between comm k and l
            tot_trials[np.diag_indices_from(tot_trials)] = self.n ** 2 / 2.  # possible connections to comm [l,l] is smaller

            ttm = np.ma.masked_values(tot_trials, 0)
            em = np.ma.masked_array(m.edge_cts, ttm.mask)

            # count each community pair only once
            unique_pairs = np.triu_indices(self.n_comm)

            t_upp.append(ttm[unique_pairs].compressed())
            e_upp.append(em[unique_pairs].compressed())
            n_graphs.append(m.n_graphs)

        # update kappa
        # key observation is that if e_lm[t] is total number of edges between comms l and m in graph type t then
        # e_lm ~ NB(kap, 1/(1+lam/(n_lm*n_graph[t])))
        # so we can use augmented conjugate update of Zhou&Carin 2012
        ps = t_upp[0]*n_graphs[0] / (t_upp[0]*n_graphs[0] + lam)
        m = e_upp[0]

        for t in range(1,np.alen(t_upp)):
            np.append(ps,t_upp[t]*n_graphs[t] / (t_upp[t]*n_graphs[t] + lam))
            np.append(ps,e_upp[t])

        kap = samp_shape_post_step(m, kap, ps, 0.1, 0.1)
        lam = mdcsbm_samp_rate_post_step(e_upp, t_upp, n_graphs, kap, lam)

        self.lam = lam
        self.kap = kap
        for m in self.agg_models:
            m.lam_base = lam
            if m.n_graphs == 0:
                m.lam = m.lam_base
            else:
                m.lam = lam / m.n_graphs
            m.kap_base = kap
            m.kap = kap