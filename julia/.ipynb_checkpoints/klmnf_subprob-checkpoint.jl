using Distributed
using Printf, LinearAlgebra, SparseArrays, Distributions
using DataFrames, StatsBase, Random, PyPlot
using RCall, LowRankApprox, Distributed
using Printf, LinearAlgebra, LowRankApprox, SparseArrays
using DataFrames, PyPlot, StatsBase, Random
using Images, FileIO, RCall

function tic()
    t0 = time_ns()
    task_local_storage(:TIMERS, (t0, get(task_local_storage(), :TIMERS, ())))
    return t0
end

function toq()
    t1 = time_ns()
    timers = get(task_local_storage(), :TIMERS, ())
    if timers === ()
        error("`toc()` without `tic()`")
    end
    t0 = timers[1]::UInt64
    task_local_storage(:TIMERS, timers[2])
    (t1-t0)/1e9
end
e = 1e-12
function eval_f(x)
    return sum(log.(W * x .+ e))
end

# https://github.com/stephenslab/mixsqp-paper/tree/master/data
W = Matrix{Float64}(readtable("simdata-n=2000-m=20.csv.gz", header = false));
# W = Matrix{Float64}(readtable("simdata-n=2000-m=200.csv.gz", header = false));
# W = Matrix{Float64}(readtable("simdata-n=20000-m=20.csv.gz", header = false));
# W = Matrix{Float64}(readtable("simdata-n=20000-m=200.csv.gz", header = false));
n,m = size(W);

o1 = zeros(iter,10)
t1 = zeros(10)
for j = 1:10
    seed = 2010 + j
    Random.seed!(seed)
    h1 = rand(m);
    h1 = h1 / sum(h1);
    tic();
    for i = 1:iter
        h1 = h1 .* (W'*(v ./ (W*h1 .+ e)))
        o1[i,j] = eval_f(h1)
    end
    t1[j] = toq()
   
end

o2 = zeros(iter,10)
t2 = zeros(10)
for j = 1:10
    seed = 2010 + j
    Random.seed!(seed)
    h2 = rand(m);
    h2 = h2 / sum(h2);
    tic();
    for i = 1:iter
        h2 = h2 .* (W'*(v ./ (W*h2 .+ e))).^2
        h2 = h2 / sum(h2)
        o2[i,j] = eval_f(h2)
    end
    t2[j] = toq()
end

o3 = zeros(iter,10)
t3 = zeros(10)
for j = 1:10
    seed = 2010 + j
    Random.seed!(seed)
    h3 = rand(m);
    h3 = h3 / sum(h3);
    tic();
    for i = 1:iter
        h3 = proj_simplex(h3 + 0.005 * (W'*(v ./ (W*h3 .+ e))))
        o3[i,j] = eval_f(h3)
    end
    t3[j] = toq()
end
