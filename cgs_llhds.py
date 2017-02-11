# some likelihood expressions used for the collapsed gibbs sampler

from scipy.special import gammaln

def beta_binom_llhd(succ, trials, a, b):
    """
    Computes the log likelihood for succ successes under Beta-Binomial with parameters given by other arguments
    :param succ: Number of observed successes, integer
    :param trials: Number of trials, integer
    :param a: Beta parameter, positive real
    :param b: Beta parameter, positive real
    :return: llhd of data, a non-positive real
    """
    return gammaln(trials + 1) + gammaln(succ + a) + gammaln(trials - succ + b) + gammaln(a + b) - \
           (gammaln(succ + 1) + gammaln(trials - succ + 1) + gammaln(a) + gammaln(b) + gammaln(trials + a + b))



def diri_multi_llhd(obs, alphas):
    """
    Computes the log likelihood for dirichlet-multinomial distribution with parameters given by other arguments
    :param obs: observed counts in each category, a vector of integers of length k
    :param alphas: Dirichlet parameter, a vector of positive reals of length k
    :return: llhd of data, a non-positive real
    """
    trials = sum(obs)
    return gammaln(trials+1) + gammaln(sum(alphas)) - gammaln(trials+sum(alphas)) + \
              sum([gammaln(ob + alpha) - gammaln(ob+1)-gammaln(alpha) for ob,alpha in zip(obs,alphas)])
