using LinearAlgebra

include("kernels.jl")

mutable struct GaussianProcessRegressor
    _kernel::Kernel
    _beta::Float64
    # cache the result of prior training
    _cov::AbstractArray{Float64, 2}
    _precision::AbstractArray{Float64, 2}
    _X::AbstractArray{Float64, 2}
    _t::AbstractArray{Float64, 1}
    function GaussianProcessRegressor(kernel::Kernel, beta=1.0)
        new(kernel, beta, zeros(1, 1), zeros(1, 1), zeros(1, 1), zeros(1))
    end
end

function log_likelihood(gp::GaussianProcessRegressor)
    return -0.5 * log(abs(det(gp._cov))) - 0.5 * (transpose(gp._t) * gp._precision * gp._t) - 0.5 * length(gp._t) * log(2.0 * pi)
end

"""
    fitting(gp::GaussianProcessRegressor, X::AbstractArray{Float64, 2}, t::AbstractArray{Float64, 1}, iter_max=0, learning_rate=0.1)

For training data X = [x_1,.... x_N] of size (n_feature, n_samples) and t = [t_1,... t_N] of size(n_samples)
"""
function fitting(gp::GaussianProcessRegressor, X::AbstractArray{Float64, 2}, t::AbstractArray{Float64, 1}, iter_max=0, learning_rate=0.1)
    n_samples = size(X)[2]
    n_samples == size(t)[1] || error("size does not match")
    
    gp._X = copy(X)
    gp._t = copy(t)
    E = Matrix(I, n_samples, n_samples) * 1.0
    K = kernel(gp._kernel, X, X)
    gp._cov = K + E / gp._beta
    gp._precision = inv(gp._cov)

    log_likes = [-Inf]
    # learn hyper parameter
    precision = gp._precision
    for i in 1:iter_max
        n_params = gp._kernel._n_params
        grads = derivatives(gp._kernel, X, X)
        updates = [-tr(precision * grads[n, :, :]) + transpose(t) * precision * grads[n, :, :] * precision * t for n in 1:n_params]
        for j in 1:iter_max
            update_parameters(gp._kernel, learning_rate * updates)
            K = kernel(gp._kernel, X, X)
            gp._cov = K + E / gp._beta
            gp._precision = inv(gp._cov)
            log_like = log_likelihood(gp)
            if log_like > log_likes[end]
                push!(log_likes, log_like)
                break
            else
                update_parameters(gp._kernel, -learning_rate * updates)
                learning_rate *= 0.9
            end
        end
    end
    popfirst!(log_likes)
    return log_likes
end

"""
    predict(gp::GaussianProcessRegressor, X::AbstractArray{Float64, 2}, with_error=false)

For input X = [x_1,... x_N] of size (n_feature, n_samples), return mus = [m(x_1),... m(x_N)] of size (n_feature, n_samples) and sigmas = [sigma(x_1),... sigma(x_N)] of size(n_feature, n_feature, n_samples) if with_error == true. Otherwise return mus.
"""
function predict(gp::GaussianProcessRegressor, X::AbstractArray{Float64, 2}, with_error=false)
    n_feature = size(X)[1]
    n_samples = size(X)[2]
    
    mus = zeros(1, n_samples)
    sigmas = zeros(1, 1, n_samples)

    for n in 1:n_samples
        x_n = X[:, n]
        x_n = reshape(x_n, size(x_n)[1], 1)
        k_n = kernel(gp._kernel, gp._X, x_n)
        size(k_n) == (size(gp._X)[2], 1) || error("size does not match: size(k_n) = $(size(k_n))")
        c = kernel(gp._kernel, x_n, x_n)
        size(c) == (1, 1) || error("size does not match")
        c = c .+ 1.0 / gp._beta
        mus[:, n] = transpose(k_n) * gp._precision * gp._t
        if with_error
            k_n = reshape(k_n, length(k_n))
            sigmas[:, :, n] = c .- transpose(k_n) * gp._precision * k_n
        end
    end

    if with_error
        return mus, sigmas
    else
        return mus
    end
end
