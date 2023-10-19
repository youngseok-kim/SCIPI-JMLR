# written in Julia 0.9

using DataFrames, StatsBase, LinearAlgebra, Random, SparseArrays
using PyPlot, Distributions, RDatasets, RCall, Missings
function update_phi(X, pi, mu, Sigma)
    n,d   = size(X);
    k     = length(pi);
    phi   = zeros(n, k);
    d     = size(Sigma[:,:,1],1);
    for i = 1:k
        for j = 1:n
            #phi[j,i] = -(X[j,:] - mu[i,:])' * inv(Sigma[:,:,1]) * (X[j,:] - mu[i,:])/2 -
            #           log(det(Sigma[:,:,i]))/2 - log(2*Base.pi)/2 * d;
            phi[j,i] = logpdf(MvNormal(mu[i,:], Sigma[:,:,i]), X[j,:]);
        end
    end
    temp  = maximum(phi, dims = 2);
    phi   = pi .* exp.(phi .- temp);
    llik  = sum(log.(sum(exp.(temp) .* phi, dims = 2)))
    phi   = phi ./ sum(phi, dims = 2);
   
    return phi, llik
end

function GMM_EM(X, k; maxiter = 100, seed = 1, e = 1e-8)
    Random.seed!(seed);
    n,d   = size(X);
    mu    = X[sample(1:n, k),:];
    Sigma = Array{Float64, 3}(zeros(d,d,k));
    for i = 1:k
        Sigma[:,:,i] = Diagonal(ones(d));
    end
    pi    = ones(1,k)/k;
    llik  = zeros(maxiter + 1);
    phi, llik[1] = update_phi(X, pi, mu, Sigma);
   
    for iter = 1:maxiter
        mu    = (phi' * X) ./ sum(phi, dims = 1)';
        for i = 1:k
            Sigma[:,:,i] = (X' .- mu[i,:]) * (phi[:,i] .* (X .- mu[i,:]')) / sum(phi[:,i]);
            Sigma[:,:,i] = Symmetric(Sigma[:,:,i]) + e * I;
        end
        pi                  = mean(phi, dims = 1);
        phi, llik[iter+1]   = update_phi(X, pi, mu, Sigma);
    end
   
    return Dict([(:mu, mu), (:Sigma, Sigma), (:phi, phi), (:pi, pi), (:llik, llik)])
end

function GMM_SCIPI(X, k; maxiter = 100, seed = 1, e = 1e-8)
    Random.seed!(seed);
    n,d   = size(X);
    mu    = X[sample(1:n, k),:];
    Sigma = Array{Float64, 3}(zeros(d,d,k));
    for i = 1:k
        Sigma[:,:,i] = Diagonal(ones(d));
    end
    pi    = ones(1,k)/k;
    llik  = zeros(maxiter + 1);
    phi, llik[1] = update_phi(X, pi, mu, Sigma);
   
    for iter = 1:maxiter
        mu    = (phi' * X) ./ sum(phi, dims = 1)';
        for i = 1:k
            Sigma[:,:,i] = (X' .- mu[i,:]) * (phi[:,i] .* (X .- mu[i,:]')) / sum(phi[:,i]);
            Sigma[:,:,i] = Symmetric(Sigma[:,:,i]) + e * I;
        end
        L     = phi ./ pi;
        pi    = pi .* (1/2 .+ (1 ./ (pi * L') ) * L / n).^2;
        pi    = pi / sum(pi);
        phi, llik[iter+1]   = update_phi(X, pi, mu, Sigma);
    end
   
    return Dict([(:mu, mu), (:Sigma, Sigma), (:phi, phi), (:pi, pi), (:llik, llik)])
end

R"library(mlbench); data(Ionosphere);
X = Ionosphere[,1:34]
X = matrix(as.numeric(as.matrix(X)), dim(X));"; @rget X;
X        = Matrix{Float64}(X);
res11    = zeros(10,2);
for i = 1:10
    @time out1       = GMM_EM(X, 2, seed = i, maxiter = 100);
    @time out2       = GMM_SCIPI(X, 2, seed = i, maxiter = 100);
    res11[i,:]       = [out1[:llik][end];out2[:llik][end]];
    print(i," ");
end

R"library(mlbench); data(Vehicle);
X = Vehicle[,1:18]
X = matrix(as.numeric(as.matrix(X)), dim(X));"; @rget X;
X        = Matrix{Float64}(X);
res12    = zeros(10,2);
for i = 1:10
    @time out1       = GMM_EM(X, 4, seed = i, maxiter = 100);
    @time out2       = GMM_SCIPI(X, 4, seed = i, maxiter = 100);
    res12[i,:]       = [out1[:llik][end];out2[:llik][end]];
    print(i," ");
end

R"library(mlbench); data(Glass);
X = Glass[,1:9]
X = matrix(as.numeric(as.matrix(X)), dim(X));"; @rget X;
X        = Matrix{Float64}(X);
res13    = zeros(10,2);
for i = 1:10
    @time out1       = GMM_EM(X, 6, seed = i, maxiter = 50);
    @time out2       = GMM_SCIPI(X, 6, seed = i, maxiter = 50);
    res13[i,:]       = [out1[:llik][end];out2[:llik][end]];
    print(i," ");
end

R"library(mlbench); data(HouseVotes84);
X = HouseVotes84[,-1];
X = as.matrix(X);
X = matrix(as.numeric(X), dim(X));
X[is.na(X)] = 0;"; @rget X;
X       = Matrix{Float64}(X);
res14   = zeros(10,2);
for i = 1:10
    @time out1       = GMM_EM(X, 2, seed = i, maxiter = 100);
    @time out2       = GMM_SCIPI(X, 2, seed = i, maxiter = 100);
    res14[i,:]       = [out1[:llik][end];out2[:llik][end]];
    print(i," ");
end

R"library(mlbench); data(Zoo);
X = Zoo[,1:16]
X = matrix(as.numeric(as.matrix(X)), dim(X));"; @rget X;
X        = Matrix{Float64}(X);
res15    = zeros(10,2);
for i = 1:10
    @time out1       = GMM_EM(X, 7, seed = i + 14, maxiter = 100);
    @time out2       = GMM_SCIPI(X, 7, seed = i + 14, maxiter = 100);
    res15[i,:]       = [out1[:llik][end];out2[:llik][end]];
    print(i," ");
end

R"library(mlbench); data(Sonar);
X = Sonar[,1:60]
X = matrix(as.numeric(as.matrix(X)), dim(X));
X[is.na(X)] = 0;"; @rget X;
X        = Matrix{Float64}(X);
res16    = zeros(10,2);
for i = 1:10
    @time out1       = GMM_EM(X, 2, seed = i + 2, maxiter = 100);
    @time out2       = GMM_SCIPI(X, 2, seed = i + 2, maxiter = 100);
    res16[i,:]       = [out1[:llik][end];out2[:llik][end]];
    print(i," ");
end

R"library(mlbench); data(Servo);
X = Servo[,-5]
X = matrix(as.numeric(as.matrix(X)), dim(X));
X[is.na(X)] = 0;"; @rget X;
X        = Matrix{Float64}(X);
res17    = zeros(10,2);
for i = 1:10
    @time out1       = GMM_EM(X, 51, seed = i, maxiter = 100);
    @time out2       = GMM_SCIPI(X, 51, seed = i, maxiter = 100);
    res17[i,:]       = [out1[:llik][end];out2[:llik][end]];
    print(i," ");
end


R"library(mlbench); data(BreastCancer);
X = BreastCancer[,-11]
X = matrix(as.numeric(as.matrix(X)), dim(X));
X[is.na(X)] = 0;"; @rget X;
X        = Matrix{Float64}(X);
res18    = zeros(10,2);
for i = 1:10
    @time out1       = GMM_EM(X, 2, seed = i, maxiter = 100);
    @time out2       = GMM_SCIPI(X, 2, seed = i, maxiter = 100);
    res18[i,:]       = [out1[:llik][end];out2[:llik][end]];
    print(i," ");
end


R"library(mlbench); data(PimaIndiansDiabetes);
X = PimaIndiansDiabetes[,-9]
X = matrix(as.numeric(as.matrix(X)), dim(X));
X[is.na(X)] = 0;"; @rget X;
X        = Matrix{Float64}(X);
res19    = zeros(10,2);
for i = 1:10
    @time out1       = GMM_EM(X, 2, seed = i, maxiter = 100);
    @time out2       = GMM_SCIPI(X, 2, seed = i, maxiter = 100);
    res19[i,:]       = [out1[:llik][end];out2[:llik][end]];
    print(i," ");
end


R"library(mlbench); data(Vowel);
X = Vowel[,1:10]
X = matrix(as.numeric(as.matrix(X)), dim(X));
X[is.na(X)] = 0;"; @rget X;
X        = Matrix{Float64}(X);
res20    = zeros(10,2);
for i = 1:10
    @time out1       = GMM_EM(X, 11, seed = i, maxiter = 100);
    @time out2       = GMM_SCIPI(X, 11, seed = i, maxiter = 100);
    res20[i,:]       = [out1[:llik][end];out2[:llik][end]];
    print(i," ");
end