import time
import numpy as np

#nice problems --> https://arxiv.org/pdf/1302.0125.pdf

#Classic RGD optimizer, for rates, check http://proceedings.mlr.press/v49/zhang16b.pdf
def RGD_optimizer(K,x0,L,cost,grad,exp):
    x = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    f = np.zeros((K,))
    f[0] = cost(x0)
    h = 1/L
    for k in range(K-1):
        t1 = time.time()
        x[k+1] = exp(x[k],-h*grad(x[k]))
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f

#RAGD optimizer, by Zhang and Sra (2018). Accelerated, check http://proceedings.mlr.press/v75/zhang18a/zhang18a.pdf
def RAGD_optimizer(K,x0,L,mu,cost,grad,exp,log):
    x = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    y = [0*x0 for i in range(K)]
    v = [x0 for i in range(K)]
    f = np.zeros((K,))
    a = np.zeros((K,))
    A = np.zeros((K,))
    f[0] = cost(x0)
    h = 1/L
    beta = np.sqrt(mu/L)/5
    alpha = (np.sqrt((beta**2)+4*(1+beta)*mu*h)-beta)/2
    gamma = mu*(np.sqrt(beta**2+4*(1+beta)*mu*h)-beta)/(np.sqrt(beta**2+4*(1+beta)*mu*h)+beta)
    gamma_bar = (1+beta)*gamma
    for k in range(K-1):
        t1 = time.time()        
        y[k] = exp(x[k],(alpha*gamma/(gamma+alpha*mu))*log(x[k],v[k]))
        x[k+1] = exp(y[k],-h*grad(y[k]))
        v[k+1] = exp(y[k],(((1-alpha)*gamma)/gamma_bar)*log(y[k],v[k])-(alpha/gamma_bar)*grad(y[k]))
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f


#Golden ratio line search along a geodesic, used at each iteration of RAGDsDR
def linesearch(v,x,log,exp,cost,line_search_iterations):    
    gr = ( 1 + np.sqrt(5) ) / 2
    p1 = 0
    p3 = 1
    p2 = 1/(1+gr)
    p4 = gr/(1+gr)
    val1 = cost(exp(v,p1*log(v,x)))
    val2 = cost(exp(v,p2*log(v,x)))
    val3 = cost(exp(v,p3*log(v,x)))
    val4 = cost(exp(v,p4*log(v,x)))
    for i in range(line_search_iterations):
        if val2<val4:
            p3 = p4
            val3 = val4
            p4 = p2
            val4 = val2
            p2 = p1+(p3-p1)/(1+gr)
            val2 = cost(exp(v,p2*log(v,x)))
            b = p2
        else:
            p1 = p2
            val1 = val2
            p2 = p4
            val2 = val4
            p4 = p3-(p3-p1)/(1+gr)
            val4 = cost(exp(v,p4*log(v,x)))
            b = p4
    return b

#An optimizer we propose in this paper, adaptation of https://arxiv.org/abs/1809.05895
def RAGDsDR_optimizer(K,x0,L,cost,grad,exp,log,transp,line_search_iterations):
    x = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    y = [0*x0 for i in range(K)]
    v = [x0 for i in range(K)]
    a = np.zeros((K,))
    beta = np.zeros((K,))
    A = np.zeros((K,))
    f = np.zeros((K,))
    f[0] = cost(x0)
    h = 1/L
    for k in range(K-1):
        t1 = time.time()        
        if line_search_iterations>0:
            beta[k] = linesearch(v[k],x[k],log,exp,cost,line_search_iterations)
        else:
            beta[k] = k/(k+2)
        y[k] = exp(v[k], beta[k]* log(v[k],x[k]))
        x[k+1] = exp(y[k],-h*grad(y[k]))
        a[k+1] = np.max(np.roots(np.array([1, -h, -h*A[k]])))
        A[k+1] = A[k]+a[k+1]
        v[k+1] = exp(v[k],-a[k+1]*transp(y[k],v[k],grad(y[k])))
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f, beta

#A modification of the last optimizer, with no parallel transport needed. Performance is identical.
def RAGDsDR_optimizer_no_transp(K,x0,L,cost,grad,exp,log,line_search_iterations): #an older version we had
    x = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    y = [0*x0 for i in range(K)]
    v = [x0 for i in range(K)]
    a = np.zeros((K,))
    beta = np.zeros((K,))
    A = np.zeros((K,))
    f = np.zeros((K,))
    f[0] = cost(x0)
    h = 1/L
    for k in range(K-1):
        t1 = time.time()
        if line_search_iterations>0:
            beta[k] = linesearch(v[k],x[k],log,exp,cost,line_search_iterations)
        else:
            beta[k] = k/(k+2)
        y[k] = exp(v[k], beta[k]* log(v[k],x[k]))
        x[k+1] = exp(y[k],-h*grad(y[k]))
        a[k+1] = np.max(np.roots(np.array([1, -h, -h*A[k]])))
        A[k+1] = A[k]+a[k+1]
        v[k+1] = exp(y[k],log(y[k],v[k])-a[k+1]*grad(y[k])) #changes here
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f, beta

#Riemannian linear coupling, adaptation of Allen-Zhu, Orecchia (2014) https://arxiv.org/abs/1407.1537
def Riemann_coupling(K,x0,L,lam,cost,grad,exp,log): 
    x = [x0 for i in range(K)]
    y = [x0 for i in range(K)]
    z = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    f = np.zeros((K,))
    f[0] = cost(x0)
    for k in range(K-1):
        t1 = time.time()
        a = (k+2)/(2*(lam**2)*L)
        tau = 2/(k+2)
        x[k+1] = exp(z[k],(1-tau)*log(z[k],y[k]))
        y[k+1] = exp(x[k+1],-(1/L)*grad(x[k+1]))
        z[k+1] = exp(x[k+1],log(x[k+1],z[k])-(lam**2)*a*grad(x[k+1]))
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f