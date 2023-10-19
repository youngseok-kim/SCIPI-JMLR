import numpy as np

def run_subprob_mu(L, w, seed = 1, num_iter = 10000, num_print = 500, eps = 1e-8):
    
    n,m = L.shape
    np.random.seed(seed * 2023)
    
    pi = np.random.rand(m)
    pi = pi / pi.sum()
    
    obj_func = np.zeros(num_iter)
    
    for i in range(num_iter):
        pi = pi * (L.T @ (w / (L @ pi + eps)))
        obj_func[i] = -(np.log(L @ pi + eps) * w).sum()
        if i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return pi, obj_func


def run_subprob_mu_with_normalize(L, w, seed = 1, num_iter = 10000, num_print = 500, eps = 1e-8):
    
    n,m = L.shape
    np.random.seed(seed * 2023)
    
    pi = np.random.rand(m)
    pi = pi / pi.sum()
    
    obj_func = np.zeros(num_iter)
    
    for i in range(num_iter):
        pi = pi * (L.T @ (w / (L @ pi + eps)))
        pi = pi / pi.sum()
        obj_func[i] = -(np.log(L @ pi + eps) * w).sum()
        if i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return pi, obj_func

def run_subprob_scipi(L, w, seed = 1, num_iter = 10000, num_print = 500, eps = 1e-8):
    
    n,m = L.shape
    np.random.seed(seed * 2023)
    
    pi = np.random.rand(m)
    pi = pi / pi.sum()
    
    obj_func = np.zeros(num_iter)
    
    for i in range(num_iter):
        pi = pi * np.square(L.T @ (w / (L @ pi + eps)))
        pi = pi / pi.sum()
        obj_func[i] = -(np.log(L @ pi + eps) * w).sum()
        if i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return pi, obj_func


# lots of papers about projection onto the simplex
# e.g.
# https://arxiv.org/pdf/1101.6081.pdf
# https://math.stackexchange.com/questions/3778014/matlab-python-euclidean-projection-on-the-simplex-why-is-my-code-wrong
# https://stanford.edu/~jduchi/projects/DuchiShSiCh08.html
# https://link.springer.com/article/10.1007/s10107-015-0946-6
# https://gist.github.com/mblondel/6f3b7aaad90606b98f71


def proj_simplex(v):
    u = (v > 0) * v
    u.sort()
    u = u[::-1]
    sv = u.cumsum()
    rho = np.where(u > (sv - 1) / np.arange(1, len(u)+1))[0][-1]
    theta = np.maximum(0.0, (sv[rho] - 1.0) / (rho+1))
    v = np.maximum(v - theta, 0.0)
    return v

def run_subprob_pgd(L, w, seed = 1, stepsize = 1.0, num_iter = 10000, num_print = 500, eps = 1e-8):
    
    n,m = L.shape
    np.random.seed(seed * 2023)
    
    pi = np.random.rand(m)
    pi = pi / pi.sum()
    
    obj_func = np.zeros(num_iter)
    
    for i in range(num_iter):
        pi = pi + stepsize * (L.T @ (w / (L @ pi + eps)))
        pi = proj_simplex(pi)
        obj_func[i] = -(np.log(L @ pi + eps) * w).sum()
        if i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return pi, obj_func

def check_sufficient_decrease(obj, obj_new, grad, grad_proj, alpha = 1.0):
    return obj_new - obj - alpha * (grad * grad_proj).sum()

def run_subprob_pgd_with_linesearch(L, w, seed = 1, init_stepsize = 1.0, num_linesearch = 20, alpha = 1.0, num_iter = 10000, num_print = 500, eps = 1e-8):
    
    n,m = L.shape
    np.random.seed(seed * 2023)
    
    pi = np.random.rand(m)
    pi = pi / pi.sum()
    
    obj_func = np.zeros(num_iter)
    
    for i in range(num_iter):
        stepsize = init_stepsize
        if i == 0:
            obj = -(np.log(L @ pi + eps) * w).sum()
        else:
            obj = obj_func[i-1]
        grad = -(L.T @ (w / (L @ pi + eps)))
        for j in range(num_linesearch):
            pi_temp = pi - stepsize * grad
            pi_temp = proj_simplex(pi_temp)
            grad_proj = pi_temp - pi
            obj_temp = -(np.log(L @ pi_temp + eps) * w).sum()
            if check_sufficient_decrease(obj, obj_temp, grad, grad_proj, alpha) < 0.0:
                break
            else:
                stepsize = stepsize * 0.5
        #print(i, stepsize)
        pi = pi - stepsize * grad
        pi = proj_simplex(pi)
        obj_func[i] = -(np.log(L @ pi + eps) * w).sum()
        if i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return pi, obj_func