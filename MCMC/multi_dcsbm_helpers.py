import numpy as np


def mdcsbm_samp_rate_post_step(e_top, n_top, n_graphs, kap, lam, steps=20):
    """
    m-h sample from p(lam | e,n,kap) for MDCSBM model where all graphs are
    presumed to have a common lambda and kappa; the derivation of the update is
    straightforward from the equivalent sampler for DCSBM

    :param e: length n_models list of nparray; e[t] is contents of upper triangular array of e[l,m] where e_lm is number of edges between comm l and m
    :param n: length n_models list of nparray; n[t] contents are upper triangular array of n[l,m], with n[l,m] = n_l*n_m for l \neq m, n[l,l] = n[l]**2/2
    :param n_graphs: length n_models list of integers; n_graphs[t] is number of graphs of model type t
    :param kap: npfloat
    :param lam: npfloat
    :param steps: int, number of mh steps
    :return: appx sample from p(lam | e,n,kap)
    """

    n_models = np.alen(n_graphs)

    z=np.random.normal(0,0.1,steps)
    lam_last = lam

    # a little bit of algebra gets the update equation down to this, which avoids overflow errors
    l_last = -np.log(lam_last) \
             + np.sum([dcsbm_lam_llhd(e_top[j], n_top[j], kap, lam_last / n_graphs[j]) for j in range(n_models)])
    for step in range(steps):
        lam_prop = np.exp(np.log(lam_last)+z[step])
        l_prop = -np.log(lam_prop) \
                 + np.sum([dcsbm_lam_llhd(e_top[j], n_top[j], kap, lam_prop / n_graphs[j]) for j in range(n_models)])

        if l_prop - l_last > 0:
            accept=1
        elif l_prop - l_last < -15:
            accept = 0
        else:
            accept = np.random.binomial(1,min(1,np.exp(l_prop - l_last)))

        if accept==1:
            lam_last = lam_prop
            l_last = l_prop

    return lam_last

def dcsbm_lam_llhd(e_top, n_top, kap, lam):
    """
    Part of the log llhd of the DCSBM that contains lambda (for MH updates of lam)
    :param e_top: nparray, vector; contents are upper triangular array of e[l,m] where e_lm is number of edges between comm l and m
    :param n_top: nparray, vector; contents are upper triangular array of n[l,m], with n[l,m] = n_l*n_m for l \neq m, n[l,l] = n[l]**2/2
    :param kap: npfloat
    :param lam: npfloat
    :param steps: int, number of mh steps
    :return: npfloat
    """
    return -np.sum((kap+e_top)*np.log(n_top+lam) - kap*np.log(lam))