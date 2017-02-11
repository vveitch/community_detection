# community_detection
Code relating to community detection in graphs

This is meant to hold code for various approaches to fiting stochastic block models and their many generalizations. That will usually (maybe always?) mean (approximate) Bayesian inference. The twin goals here are to release good out of the box tools for graph fitting---these are surprisingly lacking!---and to give myself a sandbox to play around with ideas in inference. 

----

Degree Corrected Stochastic Block Model:
First upload is a collapsed gibbs sampler for the Bayesian degree corrected stochastic block model. This is more or less the model and scheme described in https://arxiv.org/abs/1311.2520. Performance seems to be quite good for moderate number of vertices, but convergence is probably too slow past a couple thousand. 
Differences with the DC-SBM described above are that I use a different scheme for updating the negative binomial parameters (based on https://arxiv.org/abs/1209.3442), I've introduced a bunch of computational tricks (which I derived more for the exercise of working things out then because they're really necessary), and I swapped the dirichlet process to a dirichlet distribution.

One can recover a (Poisson) stochastic block model by setting the parameter gamma to be very large; but don't actually do this---the degree corrected SBM should perform much better.
