using LinearAlgebra

mutable struct GaussianFeature
    _means::Array{Float64, 1} # the list of means
    _var::Float64 # the variance of gaussian
    _n_kernels::Int64 # cache the number of kernels
    function GaussianFeature(means::Array{Float64, 1}, var::Float64)
        new(means, var, size(means)[1]);
    end
end

function _gauss(mean::Float64, var::Float64, x::Array{Float64, 1})
    return exp.(-(0.5 / var) .* (x .- mean).^2)
end

function transform(feature::GaussianFeature, x::Array{Float64, 1})
    # this feature transforms R^1 to R^1
    # x is an array of size n_samples
    # returns Phi = [phi_1(x),,,, phi_N(x)] where phi_i(x) is a gaussian centered around means[i]
    n_samples = size(x)[1];
    n_kernels = feature._n_kernels;
    Phi = zeros(n_kernels, n_samples);
    for i in 1:n_kernels
        mean = feature._means[i];
        var = feature._var;
        Phi[i, :] = _gauss(mean, var, x);
    end

    return Phi;
end
