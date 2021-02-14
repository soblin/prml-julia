using LinearAlgebra
using Distributions

mutable struct BayesianRegressor
    _alpha::Float64
    _beta::Float64
    # prior/posterior distribution N(w | w_mean, w_precision^(-1))
    _w_mean::AbstractArray{Float64, 1}
    _w_precision::AbstractArray{Float64, 2}
    function BayesianRegressor(alpha::Float64, beta::Float64, n_features::Int64)
        w_mean = zeros(n_features);
        w_precision = alpha * Matrix(I, n_features, n_features);
        new(alpha, beta, w_mean, w_precision);
    end
end

function fitting(regressor::BayesianRegressor, Phi::AbstractArray{Float64, 2}, t::AbstractArray{Float64, 1})
    # Phi is the design matrix [phi(x_1), phi(x_2), ,,, phi(x_N)]^T of size (n_samples, n_features)
    n_features = size(Phi)[2];
    n_samples = size(Phi)[1];
    @assert n_samples == size(t)[1]

    w_0 = regressor._w_mean;
    S_0_inv = regressor._w_precision;
    beta = regressor._beta;
    
    tmp = S_0_inv + beta * transpose(Phi) * Phi;
    # make S_N_inv Cholesky
    S_N_inv = (tmp + transpose(tmp)) / 2.0;
    w_N = inv(S_N_inv) * (S_0_inv * w_0 + beta * transpose(Phi) * t);

    # update
    regressor._w_mean = w_N;
    regressor._w_precision = S_N_inv;
end

function predict(regressor::BayesianRegressor, Phi::AbstractArray{Float64, 2}, return_std::Bool=false)
    # Phi is the design matrix [phi(x_1), phi(x_2), ,,, phi(x_N)]^T of size (n_samples, n_features)
    N = size(Phi)[1];
    w_mean = regressor._w_mean;
    w_cov = inv(regressor._w_precision);

    y = Phi * w_mean;

    if return_std == true
        y_var = 1.0 / regressor._beta .+ [ transpose(Phi[i, :]) * w_cov * Phi[i, :] for i in 1:N];
        y_std = sqrt.(y_var);
        return y, y_std;
    end

    return y
end

function predictSampling(regressor::BayesianRegressor, Phi::AbstractArray{Float64, 2}, n_sampling::Int64)
    # Phi is the design matrix [phi(x_1), phi(x_2), ,,, phi(x_N)]^T of size (n_samples, n_features)
    # returns [y(x_1), y(x_2),,,, y(x_N)], each y(x_i) is a AbstractArray of size n_sampling
    N = size(Phi)[1];
    ret = zeros(n_sampling, N);

    w_mean = regressor._w_mean;
    w_cov = inv(regressor._w_precision);

    w_samples = rand(MvNormal(w_mean, w_cov), n_sampling);
    for i in 1:n_sampling
        ret[i, :] = transpose(w_samples[:, i]) * transpose(Phi);
    end

    return ret;
end
