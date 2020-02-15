# Practical Accelerated Optimization on Riemannian Manifolds

arvix version: https://arxiv.org/abs/2002.04144

We develop a new Riemannian descent algorithm with an accelerated rate of convergence.  We focus on functions that are geodesically convex or weakly-quasi-convex, which are weaker function classes compared to prior work that has considered geodesically strongly convex functions. Ourproof of convergence relies on a novel estimates equence which allows to demonstrate the dependency of the convergence rate on the curvature of the manifold. We validate our theoretical results empirically on several optimization problems defined on a sphere and on the manifold of positive definite matrices.

## Requirements
pymanopt==0.2.4

torch==1.0.0

scipy==1.1.0 (statsmodels fails with scipy >=1.3.0 )

numpy==1.15.4

matplotlib==2.2.3


## Contents of this folder
This folder contains 5 files

1) geometric_optimizers.py : implementation of Riemannian Gradient Descent, Riemannian Accelerated Gradient Descent and RAGDsDR (method in this paper);

2) three notebooks containing the implementations for operator scaling, Rayleight quotient and Karcher mean (these problems are described in the notebooks); 

2) utils.py : additional functions for the problems above.

