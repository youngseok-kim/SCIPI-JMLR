using Distributed
using Printf, LinearAlgebra, SparseArrays, Distributions
using DataFrames, StatsBase, Random, PyPlot
using RCall, LowRankApprox, Distributed
using Printf, LinearAlgebra, LowRankApprox, SparseArrays
using DataFrames, PyPlot, StatsBase, Random
using Images, FileIO, RCall

function init_sparse(V, k, ind, seed)
    Random.seed!(seed)
    n,m    = size(V);
    W      = rand(n,k);
    H      = rand(k,m);
    A      = zeros(n,m);
    A[ind]  = V.nzval ./ (W * H)[ind];
    H       = H .* (W' * A) ./ sum(W, dims = 1)';
    A[ind]  = V.nzval ./ (W * H)[ind];
    W       = W .* (A * H') ./ sum(H, dims = 2)';
    H       = H ./ sum(H .+ e, dims = 2);
    W       = W ./ sum(W .+ e, dims = 1);
   
    return W, H, A
end

function KLNMF_EM(V, k; seed = 1, iter = 1000)
    Random.seed!(1);
    W, H, A = init_sparse(V,k,ind, seed);
    obj1    = zeros(iter);
    a       = 1000 * log(1000) - 999;
    for i = 1:iter
        if i == 1
        else
            H       = H .* (W' * A) ./ sum(W, dims = 1)';
            A[ind]  = V.nzval ./ (W * H)[ind];
            W       = W .* (A * H') ./ sum(H, dims = 2)';
            A[ind]  = V.nzval ./ (W * H)[ind];
        end
        #WH      = W * H / sum(W * H);
        obj1[i] = a + dot(V.nzval, log.(A[ind]));
        #obj1[i] = -dot(V, log.(WH .+ e)) + dot(V, log.(V .+ e)) - sum(V) + sum(WH);
        #obj2[i] = dot(V, log.(WH .+ e));
        print(i, " ");
    end
    return W, H, A, obj1
end

function KLNMF_SCIPI(V, k; seed = 1, iter = 1000, b = 0)
    Random.seed!(1);
    W, H, A = init_sparse(V,k,ind, seed);
    obj1    = zeros(iter);
    a       = 1000 * log(k) - 999;
    for i = 1:iter
        if i == 1
        else
            H       = H .* (b .+ W' * A).^2;
            H       = H ./ sum(H .+ e, dims = 2);
            A[ind]  = V.nzval ./ (W * H)[ind];
            W       = W .* (b .+ A * H').^2;
            W       = W ./ sum(W .+ e, dims = 1);
            A[ind]  = V.nzval ./ (W * H)[ind];
        end
        obj1[i] = a + dot(V.nzval, log.(A[ind]));
        print(i, " ");
    end
    return W, H, A, obj1
end

function KLNMF_SCIPI1(V, k; seed = 1, iter = 1000, b = 1)
    Random.seed!(1);
    W, H, A = init_sparse(V,k,ind, seed);
    obj1    = zeros(iter);
    a       = 1000 * log(k) - 999;
    for i = 1:iter
        if i == 1
        else
            H       = H .* (b .+ W' * A).^2;
            H       = H ./ sum(H .+ e, dims = 2);
            A[ind]  = V.nzval ./ (W * H)[ind];
            W       = W .* (b .+ A * H').^2;
            W       = W ./ sum(W .+ e, dims = 1);
            A[ind]  = V.nzval ./ (W * H)[ind];
        end
        obj1[i] = a + dot(V.nzval, log.(A[ind]));
        print(i, " ");
    end
    return W, H, A, obj1
end

function KLNMF_PGD(V, k; seed = 1, iter = 1000, stepsize = 1.1)
    Random.seed!(1);
    W, H, A = init_sparse(V,k,ind, seed);
    obj1    = zeros(iter);
    for i = 1:iter
        if i == 1
        else
            H       = H + (W' * A .- sum(W, dims = 1)') .* H ./ sum(W, dims = 1)' * stepsize;
            H       = max.(H, 0);
            A[ind]  = V.nzval ./ (W * H)[ind];
            W       = W + (A * H' .- sum(H, dims = 2)') .* W ./ sum(H, dims = 2)' * stepsize;
            W       = max.(W, 0);
            A[ind]  = V.nzval ./ (W * H)[ind];
        end
        #H       = H ./ sum(H .+ e, dims = 2);
        #W       = W ./ sum(W .+ e, dims = 1);
        obj1[i] = 1000 * log(sum(W * H)) + dot(V.nzval, log.(A[ind])) - 999;
        print(i, " ");
    end
    return W, H, A, obj1
end

function KLNMF_SCIPI_INNER(V, k; seed = 1, inneriter = 10, iter = 1000, b = 0)
    Random.seed!(1);
    W, H, A = init_sparse(V,k,ind, seed);
    obj1    = zeros(iter);
    a       = 1000 * log(k) - 999;
    for i = 1:iter
        if i == 1
        else
            for j = 1:inneriter
                H       = H .* (b .+ W' * A).^2;
                H       = H ./ sum(H .+ e, dims = 2);
                A[ind]  = V.nzval ./ (W * H)[ind];
            end
            for j = 1:inneriter
                W       = W .* (b .+ A * H').^2;
                W       = W ./ sum(W .+ e, dims = 1);
                A[ind]  = V.nzval ./ (W * H)[ind];
            end
        end
        obj1[i] = a + dot(V.nzval, log.(A[ind]));
        print(i, " ");
    end
    return W, H, A, obj1
end

V  = readtable("docword.kos.txt", separator = ' ', header = false);
X  = Matrix(V);
m  = maximum(X[:,1]);
n  = maximum(X[:,2]);
V  = zeros(n,m);
for i = 1:size(X,1)
    V[X[i,2],X[i,1]] = X[i,3]
end
V = sparse(V);
ind  = findall(V .> 0);
n,m = size(V);
e = 1e-12;
V = V / sum(V) * 1e3;

using CSV
CSV.write("kos_fullres2.txt", DataFrame(out1), delim = ',')

out1 = zeros(10,2700)
for i = 1:10
    @time oo1 = KLNMF_EM(V, 20; seed = 2010 + i, iter = 800);
    @time oo2 = KLNMF_SCIPI(V, 20; seed = 2010 + i, iter = 800);
    @time oo3 = KLNMF_PGD(V, 20; seed = 2010 + i, iter = 800);
    @time oo4 = KLNMF_SCIPI_INNER(V, 20; seed = 2010 + i, iter = 300);
    out1[i,:]  = [oo1[4];oo2[4];oo3[4];oo4[4]]
end

X = readtable("wiki-Vote.txt", separator = ' ', header = false);
V = sparse(X[1],X[2],1);
ind  = findall(V .> 0);
n,m = size(V);
e = 1e-12;
V = V / sum(V) * 1e3;

out2 = zeros(10,3 * 1700)
for i = 1:10
    @time oo1 = KLNMF_EM(V, 20; seed = 2010 + i, iter = 500);
    @time oo2 = KLNMF_SCIPI(V, 20; seed = 2010 + i, iter = 500);
    @time oo3 = KLNMF_PGD(V, 20; seed = 2010 + i, iter = 500);
    @time oo4 = KLNMF_SCIPI_INNER(V, 20; seed = 2010 + i, iter = 200);
    out2[i,:]  = [oo1[4];oo2[4];oo3[4];oo4[4]]
end

files = readdir("WavingTrees");
X = zeros(120 * 160,287);
for i = 2:length(files)
    img      = load("WavingTrees/" * files[i]);
    temp     = Matrix{Float64}(Gray.(img));
    X[:,i-1] = temp[:];
end
V    = sparse(X);
ind  = findall(V .> 0);
n,m = size(V);
e = 1e-12;
V = V / sum(V) * 1e3;

out3 = zeros(10,3 * 1700)
for i = 1:10
    @time oo1 = KLNMF_EM(V, 20; seed = 2010 + i, iter = 500);
    @time oo2 = KLNMF_SCIPI(V, 20; seed = 2010 + i, iter = 500);
    @time oo3 = KLNMF_PGD(V, 20; seed = 2010 + i, iter = 500);
    @time oo4 = KLNMF_SCIPI_INNER(V, 20; seed = 2010 + i, iter = 200);
    out3[i,:]  = [oo1[4];oo2[4];oo3[4];oo4[4]]
end

V  = readtable("docword.nips.txt", separator = ' ', header = false);
X  = Matrix(V);
m  = maximum(X[:,1]);
n  = maximum(X[:,2]);
V  = zeros(n,m);
for i = 1:size(X,1)
    V[X[i,2],X[i,1]] = X[i,3]
end
V = sparse(V);
ind  = findall(V .> 0);
n,m = size(V);
e = 1e-12;
V = V / sum(V) * 1e3;

out4 = zeros(10, 2700)
for i = 1:10
    @time oo1 = KLNMF_EM(V, 20; seed = 2010 + i, iter = 800);
    @time oo2 = KLNMF_SCIPI(V, 20; seed = 2010 + i, b = 1, iter = 800);
    @time oo3 = KLNMF_PGD(V, 20; seed = 2010 + i, iter = 800);
    @time oo4 = KLNMF_SCIPI_INNER(V, 20; seed = 2010 + i, iter = 300);
    out4[i,:]  = [oo1[4];oo2[4];oo3[4];oo4[4]]
end