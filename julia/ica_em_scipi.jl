using DataFrames, StatsBase, LinearAlgebra, Random, SparseArrays
using Distributions, DelimitedFiles, Printf

function SCI_PI(X, w_PI; maxiter = 100, tolerance = 10^(-8))

    N,d   = size(X);
    w_PI_old = zeros(d,1);

    iter = 0

    while (min(norm(w_PI-w_PI_old, Inf),norm(w_PI+w_PI_old, Inf)) > tolerance) && (iter < maxiter)

       w_PI_old = w_PI;

       w_PI = 4 * X' * (((X*w_PI).^4 - 3*ones(N,1)) .* (X * w_PI).^3);
       w_PI = w_PI/norm(w_PI,2);

    end

    obj_PI = sum(((X*w_PI).^4 - 3*ones(N,1)).^2)

    return w_PI, obj_PI
end

function fastICA(X, w_fastICA; maxiter = 100, tolerance = 10^(-8))

    N,d   = size(X);
    w_fastICA_old = zeros(d,1);

    iter = 0

    while (min(norm( w_fastICA- w_fastICA_old, Inf),norm( w_fastICA+ w_fastICA_old, Inf)) > tolerance) && (iter < maxiter)

       w_fastICA_old = w_fastICA;

       w_fastICA = 4*X'*(X*w_fastICA).^3 - 12*sum((X*w_fastICA).^2)*w_fastICA;
       w_fastICA = w_fastICA/norm(w_fastICA,2);

    end

    obj_fastICA = sum(((X*w_fastICA).^4 - 3*ones(N,1)).^2)

    return w_fastICA, obj_fastICA
end


data_path = @sprintf("data_list.txt")
file_list = readdlm(data_path, ' ', String)

seed = 1;

for i = 1 : length(file_list)

    # Read Data
    dname = file_list[i];
    X  = Matrix{Float64}(readtable(@sprintf("dataset/%s.csv",dname), separator = ',', header = false));

    # Initialization
    Random.seed!(seed);

    # Data Preprocessing
    N,d   = size(X);

    for i = 1 : N
        X[i,:] = X[i,:] - mean(X, dims = 1)';
    end

    C = X'*X/N;
    D, V = eigen(C)

    for i = 1 : N
        X[i,:] = (V * Diagonal(D.^(-1/2)') * V' * X[i,:])';
    end

    w_0 = ones(d,1);
    w_0 = w_0/norm(w_0,2);

    w_PI, obj_PI = SCI_PI(X,w_0)
    w_fastICA, obj_fastICA = fastICA(X,w_0)

    print(obj_fastICA)
end