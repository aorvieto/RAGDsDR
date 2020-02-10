# Practical Accelerated Optimization on Riemannian Manifolds

We develop a new Riemannian descent algorithmwith an accelerated rate of convergence.  We focus on functions that are geodesically convex or weakly-quasi-convex, which are weaker function classes compared to prior work that has considered geodesically strongly convex functions. Ourproof of convergence relies on a novel estimatesequence which allows to demonstrate the dependency of the convergence rate on the curvature ofthe manifold. We validate our theoretical resultsempirically on several optimization problems de-fined on a sphere and on the manifold of positivedefinite matrices.


## Contents of this folder
This folder contains 5 files

1) geometric_optimizers.py : implementation of Riemannian Gradient Descent, Riemannian Accelerated Gradient Descent and RAGDsDR (method in this paper);

2) 3 notebooks containing the implementation of operator scaling, Rayleight quotient and Karcher mean; 

2) utils.py : additional functions for the problems above.

