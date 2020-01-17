import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm, logm, expm

def random_psd(n,kappa,seed):
    np.random.seed(seed)
    W=np.random.rand(n,n)-np.random.rand(n,n)
    X=np.dot(W.T,W)
    X=X-(np.identity(n)*np.min(la.eig(X)[0]))
    X=X/la.norm(X,2)
    X=X+np.identity(n)/(kappa-1)
    A=X/la.norm(X,2)
    return A

def karcher_mean(A, tol):
    n = A.shape[-1]
    x_sol = np.mean(A,axis=2)
    error = 1
    while error>tol:
        sqrt_x_sol = la.cholesky(x_sol)
        av_logs = 0*x_sol
        for i in range(n): av_logs+=(1/n)*logm(la.multi_dot([sqrt_x_sol.T,la.inv(A[:,:,i]),sqrt_x_sol]))
        x_sol_new = la.multi_dot([sqrt_x_sol,expm(-(1/5)*av_logs),sqrt_x_sol.T])
        error = la.norm(x_sol_new-x_sol,"fro")
        print(error)
        x_sol = x_sol_new
    return x_sol
    
        
