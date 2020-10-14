# Momentum Improves Optimization on Riemannian Manifolds


We develop a new Riemannian descent algorithm that relies on momentum to improve over existing first-order methods for geodesically convex optimization. In contrast, accelerated convergence rates proved in prior work have only been shown to hold for geodesically strongly-convex objective functions. We further extend our algorithm to geodesically weakly-quasi-convex objectives. Our proofs of convergence rely on a novel estimate sequence that illustrates the dependency of the convergence rate on the curvature of the manifold. We validate our theoretical results empirically on several optimization problems defined on the sphere and on the manifold of positive definite matrices.

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

