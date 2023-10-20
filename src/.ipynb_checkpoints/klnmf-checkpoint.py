import numpy as np

def init_klnmf(V, k, seed = 1, our_dtype = 'float32', eps = 1e-8):
    
    np.random.seed(seed)
    n,m = V.shape
    W = np.random.rand(n, k).astype(our_dtype)
    H = np.random.rand(k, m).astype(our_dtype)
    A = V / (W @ H)    
    H = H * (W.T @ A) / W.sum(axis = 0, keepdims = True).T
    A = V / (W @ H)    
    W = W * (A @ H.T) / H.sum(axis = 1, keepdims = True).T
    
    H = H / H.sum(axis = 1, keepdims = True)
    W = W / W.sum(axis = 0, keepdims = True)
    return W, H, A

def run_mu(V, k, seed = 1, num_iter = 1000, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            #A = V / (W @ H)
            H = H * (W.T @ A) / W.sum(axis = 0, keepdims = True).T
            #
            A = V / (W @ H)
            W = W * (A @ H.T) / H.sum(axis = 1, keepdims = True).T
            #
            A = V / (W @ H)
        else:
            A = V / (W @ H)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func


def run_mu_for_sparse(V, k, seed = 1, num_iter = 1000, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            #A = V / (W @ H)
            H = H * (W.T @ A) / W.sum(axis = 0, keepdims = True).T
            #
            A = V / (W @ H + eps)
            W = W * (A @ H.T) / H.sum(axis = 1, keepdims = True).T
            #
            A = V / (W @ H + eps)
        else:
            A = V / (W @ H + eps)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func
    
    
def run_mu_with_normalize(V, k, seed = 1, num_iter = 1000, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            #A = V / (W @ H)
            H = H * (W.T @ A) / W.sum(axis = 0, keepdims = True).T
            H = H / H.sum(axis = 1, keepdims = True)
            #
            A = V / (W @ H)
            W = W * (A @ H.T) / H.sum(axis = 1, keepdims = True).T
            W = W / W.sum(axis = 0, keepdims = True)
            #
            A = V / (W @ H)
        else:
            A = V / (W @ H)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func


def run_mu_with_normalize_for_sparse(V, k, seed = 1, num_iter = 1000, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            #A = V / (W @ H)
            H = H * (W.T @ A) / W.sum(axis = 0, keepdims = True).T
            H = H / H.sum(axis = 1, keepdims = True)
            #
            A = V / (W @ H + eps)
            W = W * (A @ H.T) / H.sum(axis = 1, keepdims = True).T
            W = W / W.sum(axis = 0, keepdims = True)
            #
            A = V / (W @ H + eps)
        else:
            A = V / (W @ H + eps)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func
    
    
def run_scipi(V, k, seed = 1, num_iter = 1000, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            # A = V / (W @ H)
            H = H * np.square(W.T @ A)
            H = H / H.sum(axis = 1, keepdims = True)
            #
            A = V / (W @ H)
            W = W * np.square(A @ H.T)
            W = W / W.sum(axis = 0, keepdims = True)
            #
            A = V / (W @ H)
        else:
            A = V / (W @ H)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func


def run_scipi_for_sparse(V, k, seed = 1, num_iter = 1000, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            # A = V / (W @ H)
            H = H * np.square(W.T @ A)
            H = H / H.sum(axis = 1, keepdims = True)
            #
            A = V / (W @ H + eps)
            W = W * np.square(A @ H.T)
            W = W / W.sum(axis = 0, keepdims = True)
            #
            A = V / (W @ H + eps)
        else:
            A = V / (W @ H + eps)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func


def run_scipi_acc(V, k, seed = 1, num_iter = 1000, num_inner = 2, intercept = 0.0, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            # A = V / (W @ H)
            for j in range(num_inner):
                H = H * np.square(W.T @ A + intercept)
                H = H / H.sum(axis = 1, keepdims = True)
                #
                A = V / (W @ H)
            for j in range(num_inner):
                W = W * np.square(A @ H.T + intercept)
                W = W / W.sum(axis = 0, keepdims = True)
                #
                A = V / (W @ H)
        else:
            A = V / (W @ H)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func

def run_scipi_acc_for_sparse(V, k, seed = 1, num_iter = 1000, num_inner = 2, intercept = 0.0, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            # A = V / (W @ H)
            for j in range(num_inner):
                H = H * np.square(W.T @ A + intercept)
                H = H / H.sum(axis = 1, keepdims = True)
                #
                A = V / (W @ H + eps)
            for j in range(num_inner):
                W = W * np.square(A @ H.T + intercept)
                W = W / W.sum(axis = 0, keepdims = True)
                #
                A = V / (W @ H + eps)
        else:
            A = V / (W @ H + eps)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func
    
    
def run_pgd(V, k, seed = 1, num_iter = 1000, stepsize = 1.0, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            # A = V / (W @ H)
            H = H + (W.T @ A - W.sum(axis = 0, keepdims = True).T) * H / W.sum(axis = 0, keepdims = True).T * stepsize
            H = np.maximum(H, 0.0)
            #
            A = V / (W @ H)
            W = W + (A @ H.T - H.sum(axis = 1, keepdims = True).T) * W / H.sum(axis = 1, keepdims = True).T * stepsize
            W = np.maximum(W, 0.0)
            #
            A = V / (W @ H)
        else:
            A = V / (W @ H)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func


def run_pgd_for_sparse(V, k, seed = 1, num_iter = 1000, stepsize = 1.0, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            # A = V / (W @ H)
            H = H + (W.T @ A - W.sum(axis = 0, keepdims = True).T) * H / W.sum(axis = 0, keepdims = True).T * stepsize
            H = np.maximum(H, 0.0)
            #
            A = V / (W @ H + eps)
            W = W + (A @ H.T - H.sum(axis = 1, keepdims = True).T) * W / H.sum(axis = 1, keepdims = True).T * stepsize
            W = np.maximum(W, 0.0)
            #
            A = V / (W @ H + eps)
        else:
            A = V / (W @ H + eps)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func
    
    
def check_sufficient_decrease(obj, obj_new, grad, grad_proj, alpha = 1.0):
    return obj_new - obj - alpha * (grad * grad_proj).sum()
    
def run_pgd_with_linesearch(V, k, seed = 1, num_iter = 1000, init_stepsize = 1.0, num_linesearch = 10, alpha = 1.0, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            obj = obj_func[i-1]
            G = -(W.T @ A - W.sum(axis = 0, keepdims = True).T) * H / W.sum(axis = 0, keepdims = True).T
            stepsize = init_stepsize
            for j in range(num_linesearch):
                H_temp = H - G * stepsize
                H_temp = np.maximum(H_temp, 0.0)
                G_proj = H_temp - H
                A_temp = V / (W @ H_temp)
                obj_temp = obj_intercept + (V * np.log(A_temp + eps)).sum().sum()
                if check_sufficient_decrease(obj, obj_temp, G, G_proj, alpha) < 0.0:
                    break
                else:
                    stepsize = stepsize * 0.5
            #print(i, stepsize)
            H = H - G * stepsize
            H = np.maximum(H, 0.0)
            A = V / (W @ H)
            obj = obj_intercept + (V * np.log(A + eps)).sum().sum()
            #
            G = -(A @ H.T - H.sum(axis = 1, keepdims = True).T) * W / H.sum(axis = 1, keepdims = True).T
            W = W - G * stepsize
            stepsize = init_stepsize
            for j in range(num_linesearch):
                W_temp = W - G * stepsize
                W_temp = np.maximum(W_temp, 0.0)
                G_proj = W_temp - W
                A_temp = V / (W_temp @ H)
                obj_temp = obj_intercept + (V * np.log(A_temp + eps)).sum().sum()
                if check_sufficient_decrease(obj, obj_temp, G, G_proj, alpha) < 0.0:
                    break
                else:
                    stepsize = stepsize * 0.5
            #print(i, stepsize)
            W = W - G * stepsize
            W = np.maximum(W, 0.0)
            #
            A = V / (W @ H)
        else:
            A = V / (W @ H)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func

def run_pgd_with_linesearch_for_sparse(V, k, seed = 1, num_iter = 1000, init_stepsize = 1.0, num_linesearch = 10, alpha = 1.0, num_print = 50, eps = 1e-8):
    
    W, H, A = init_klnmf(V, k, seed = seed, eps = eps)
    
    np.random.seed(seed * 2023)
    
    obj_func = np.zeros(num_iter)
    obj_intercept = k * np.log(k) - k + 1
    
    for i in range(num_iter):
        if i > 0:
            #
            obj = obj_func[i-1]
            G = -(W.T @ A - W.sum(axis = 0, keepdims = True).T) * H / W.sum(axis = 0, keepdims = True).T
            stepsize = init_stepsize
            for j in range(num_linesearch):
                H_temp = H - G * stepsize
                H_temp = np.maximum(H_temp, 0.0)
                G_proj = H_temp - H
                A_temp = V / (W @ H_temp)
                obj_temp = obj_intercept + (V * np.log(A_temp + eps)).sum().sum()
                if check_sufficient_decrease(obj, obj_temp, G, G_proj, alpha) < 0.0:
                    break
                else:
                    stepsize = stepsize * 0.5
            #print(i, stepsize)
            H = H - G * stepsize
            H = np.maximum(H, 0.0)
            A = V / (W @ H + eps)
            obj = obj_intercept + (V * np.log(A + eps)).sum().sum()
            #
            G = -(A @ H.T - H.sum(axis = 1, keepdims = True).T) * W / H.sum(axis = 1, keepdims = True).T
            W = W - G * stepsize
            stepsize = init_stepsize
            for j in range(num_linesearch):
                W_temp = W - G * stepsize
                W_temp = np.maximum(W_temp, 0.0)
                G_proj = W_temp - W
                A_temp = V / (W_temp @ H)
                obj_temp = obj_intercept + (V * np.log(A_temp + eps)).sum().sum()
                if check_sufficient_decrease(obj, obj_temp, G, G_proj, alpha) < 0.0:
                    break
                else:
                    stepsize = stepsize * 0.5
            #print(i, stepsize)
            W = W - G * stepsize
            W = np.maximum(W, 0.0)
            #
            A = V / (W @ H + eps)
        else:
            A = V / (W @ H + eps)
        obj_func[i] = obj_intercept + (V * np.log(A + eps)).sum().sum()
        if i == 0:
            print(f"init: obj {obj_func[i]}")
        elif i % num_print == num_print-1:
            print(f"round {i+1}: obj {obj_func[i]}")
    return W, H, A, obj_func