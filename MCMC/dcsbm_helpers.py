import numpy as np
from scipy.special import gammaln as spgl

"""
Helpers for computing likelihood terms for degree corrected SBM
"""


def gammaln(n):
    n = np.asarray(n, dtype=float)
    # return n*(np.log(n)-1)+0.5*np.log(2.*np.pi / n)
    small = n<1000
    rs = spgl(n[small])
    big = np.logical_not(small)
    rb = n[big]*(np.log(n[big])-1)+0.5*np.log(2.*np.pi / n[big])
    # if output is scalar need to do this because of numpy indexing in py2.7
    if np.alen(rs) == 0:
        return rb
    if np.alen(rb) == 0:
        return rs

    ret = np.zeros(np.alen(n))
    ret[small] = rs
    ret[big] = rb
    return ret

def gammaln_diff(n_in,k_in):
    """
    Approximately computes (elementwise) ln(gamma(n+k))-ln(gamma(n))
    Uses Stirling's approximation, error (per term) is O(1/n^3), and significantly better when k<<n
    :param n: numpy vector of positive real numbers
    :param k: numpy vector of positive real numbers
    :return: positive real number, approximation for ln(gamma(n+k))-ln(gamma(n))
    """
    # return (n-0.5)*np.log(1.+k/n) + k*(np.log(n+k)-1.) - k/(12.*n*(n+k))

    # make sure we're working with nparrays
    if np.isscalar(n_in):
        n = np.array(n_in)
    else:
        n = n_in

    if np.isscalar(k_in):
        k = np.array(k_in)
    else:
        k = k_in


    eps = 0.001 # error tolerance

    # for small n
    small = np.power(n,3) < 1./eps
    ret_s = gammaln(n[small]+k[small])-gammaln(n[small])

    # approximation for big inputs
    big = np.logical_not(small)
    ret_b = (n[big]-0.5)*np.log(1.+k[big]/n[big]) + k[big]*np.log(n[big]+k[big]) - k[big]/(12.*n[big]*(n[big]+k[big]))

    # if output is scalar need to do this because of numpy indexing in py2.7
    if np.alen(ret_s)==0:
        return ret_b
    if np.alen(ret_b)==0:
        return ret_s

    ret = np.zeros(np.alen(n))
    ret[small] = ret_s
    ret[big] = ret_b
    return ret

def GD(x,y,k,l):
    """
    Computes log(G(x+k,y+l)/G(x,y)) where G(x,y) = y^-x * Gamma(x)
    :param x: nparray, float
    :param y: nparray, float
    :param k: nparray, float
    :param l: nparray, float
    :return: nparray, float
    """
    return -(x+k)*np.log(y+l)+x*np.log(y)+gammaln_diff(x,k)

def BD(x,y):
    """
    Computes log(B((x,y))/B(x)) where B(x) = \prod_i Gamma(x_i) / Gamma(\sum_i x_i)
    :param x: nparray, float
    :param y: nparray, float
    :return: float
    """
    return gammaln(y)-gammaln_diff(np.sum(x),y)

def BD2(x,y):
    """
    Computes log(B(x+y)/B(x)) where B(x) = \prod_i Gamma(x_i) / Gamma(\sum_i x_i)
    :param x: nparray, float
    :param y: nparray, float
    :return: float
    """
    if len(x)==0:
        return 0
    else:
        return np.sum(gammaln(x+y)) - np.sum(gammaln(x)) -gammaln_diff(np.sum(x),np.sum(y))




"""
Posterior sampling for higher level params
"""

def crt(m,r,eps=0.001):
    """
    (Approximate) sample from a Chinese Restaurant Table (CRT) distribution,
    from M. Zhou & L. Carin. Negative Binomial Count and Mixture Modeling

    l ~ CRT(m, r) can be sampled as the sum of indep. Bernoullis:

            l = \sum_{n=1}^m Bernoulli(r/(r + n-1))

    where m >= 0 is integer and r >=0 is real.

    When m is large, approximation via Le Cam's can result in massive speedup with tiny approximation error

    :param m: np vec, length d
    :param r: scalar, float
    :param eps, scalar, float, total variation error tolerance
    :return: np vec, length d
    """

    d = np.alen(m)

    if (eps==0):
        th = m
    else:
        th = np.ceil(np.minimum(m,1. + (1+eps)/eps*r))
        th = th.astype(int)

    probs = r/(1.*r+np.arange(np.max(th)))

    # compute the expression exactly for the prefix
    l = np.zeros(d)
    for j in range(d):
        l[j]=np.sum(np.random.binomial(1,probs[0:th[j]]))

    # Le Cam's approximation for the rest of the sum
    # approximately poisson w mean r*sum_n=th^m r/(r+n-1)
    for j in range(d):
        if (th[j] < m[j]):
            mean = r*(np.log(m[j]+r-1)-np.log(th[j]+r-1)) #approximation for harmonic sum... this is a bit sketchy
            l[j] += np.random.poisson(mean)

    return l

def samp_shape_post_step(m,r,p,r_1,c_1):
    """
    returns a single step of aux gibbs sampler for p(r | m, p)
    where r \dist gamma(r_1, c_1), m_i \dist NB(r,p_i)
    :param m: np.array, length d
    :param r: np.float
    :param p: np.array, length d
    :param r_1: np.float
    :param c_1: np.float
    :return: np.float
    """

    ls = crt(m, r) # auxilary table values
    post_shape = r_1 + np.sum(ls)
    post_rate = c_1 - np.sum(np.log(1.-p))

    return np.random.gamma(post_shape, 1./post_rate, 1)


def samp_rate_post_step(e_top, n_top, kap, lam, steps=20):
    """
    m-h sample from p(lam | e,n,kap) \propto 1/lam * G(e+kap, n_lm+lam)/G(kap,lam)
    with G defined in Bayesian DCSBM paper

    :param e_top: nparray, vector; contents are upper triangular array of e[l,m] where e_lm is number of edges between comm l and m
    :param n_top: nparray, vector; contents are upper triangular array of n[l,m], with n[l,m] = n_l*n_m for l \neq m, n[l,l] = n[l]**2/2
    :param kap: npfloat
    :param lam: npfloat
    :param steps: int, number of mh steps
    :return: appx sample from p(lam | e,n,kap)
    """

    z=np.random.normal(0,0.1,steps)
    lam_last = lam
    # l_last = -np.log(lam_last) + np.sum(GD(kap,lam_last,e_top,n_top))

    # a little bit of algebra gets the update equation down to this, which avoids overflow errors
    l_last = -np.log(lam_last) - np.sum((kap+e_top)*np.log(n_top+lam_last) - kap*np.log(lam_last))
    for step in range(steps):
        lam_prop = np.exp(np.log(lam_last)+z[step])
        # l_prop = -np.log(lam_prop) + np.sum(GD(kap,lam_prop,e_top,n_top))
        l_prop = -np.log(lam_prop) - np.sum((kap + e_top) * np.log(n_top + lam_prop) - kap * np.log(lam_prop))

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


def samp_gam_post_step(terms, comm_idxs, gam, steps=20):
    """
    m-h sample from p(gam | terms, z) with llhd defined in Bayesian DCSBM paper
    and p(gam) propto 1/gam

    :param terms: nparray, vector; terms[i] is number of termini incident on vertex i
    :param comm_idxs: length n_comm list of lists, comm_idxs[k] is list of vertices in community k
    :param gam: npfloat
    :param steps: int, number of mh steps
    :return: appx sample from p(gam | term,z)
    """

    z=np.random.normal(0,0.1,steps)
    gam_last = gam
    l_last = -np.log(gam_last) \
             + np.sum([BD2(np.repeat(gam_last,len(comm_idx)),terms[comm_idx]) for comm_idx in comm_idxs])

    for step in range(steps):
        gam_prop = np.exp(np.log(gam_last)+z[step])
        l_prop = -np.log(gam_prop) \
                 + np.sum([BD2(np.repeat(gam_prop, len(comm_idx)), terms[comm_idx]) for comm_idx in comm_idxs])

        if l_prop - l_last > 0:
            accept=1
        elif l_prop - l_last < -15:
            accept = 0
        else:
            accept = np.random.binomial(1,min(1,np.exp(l_prop - l_last)))

        if accept==1:
            gam_last = gam_prop
            l_last = l_prop

    return gam_last