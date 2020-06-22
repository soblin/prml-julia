using LinearAlgebra
using Distributions

mutable struct BayesianRegressor
    _alpha::Float64
    _beta::Float64
    # prior/posterior distribution N(w | w_mean, w_precision^(-1))
    _w_mean::Array{Float64, 1}
    _w_precision::Array{Float64, 2}
    function BayesianRegressor(alpha::Float64, beta::Float64, n_features::Int64)
        w_mean = zeros(n_features);
        w_precision = alpha * Matrix(I, n_features, n_features);
        new(alpha, beta, w_mean, w_precision);
    end
end

function fitting(regressor::BayesianRegressor, PhiT::Array{Float64, 2}, t::Array{Float64, 1})
    # PhiT is the array [phi(x_1), phi(x_2), ,,, phi(x_N)]
    n_features = size(PhiT)[1];
    n_samples = size(PhiT)[2];
    @assert n_samples == size(t)[1]

    w_0 = regressor._w_mean;
    S_0_inv = regressor._w_precision;
    beta = regressor._beta;
    
    S_N_inv = S_0_inv + beta * PhiT * transpose(PhiT);
    w_N = inv(S_N_inv) * (S_0_inv * w_0 + beta * PhiT * t);

    # update
    regressor._w_mean = w_N;
    regressor._w_precision = S_N_inv;
end

function predict(regressor::BayesianRegressor, PhiT::Array{Float64, 2}, return_std::Bool=false)
    # PhiT is the list of feature vectors [phi(x_1), phi(x_2),,, phi(x_N)]
    N = size(PhiT)[2];
    w_mean = regressor._w_mean;
    w_cov = inv(regressor._w_precision);

    y = transpose(PhiT) * w_mean;

    if return_std == true
        y_var = 1.0 / regressor._beta .+ [transpose(PhiT[:, i]) * w_cov * PhiT[:, i] for i in 1:N];
        y_std = sqrt.(y_var);
        return y, y_std;
    end

    return y
end

function predictSampling(regressor::BayesianRegressor, PhiT::Array{Float64, 2}, n_sampling::Int64)
    # PhiT is the list of feature vectors [phi(x_1), phi(x_2),,, phi(x_N)]
    # sample some predicted values for each phi(x_i)
    # returns [y(x_1), y(x_2),,,, y(x_N)], each y(x_i) is a array of size n_sampling
    # (n_sampling x N)
    N = size(PhiT)[2];
    ret = zeros(n_sampling, N);

    w_mean = regressor._w_mean;
    w_cov = inv(regressor._w_precision);

    w_samples = rand(MvNormal(w_mean, w_cov), n_sampling);
    for i in 1:n_sampling
        ret[i, :] = transpose(w[:, i]) * PhiT;
    end

    return ret;
end
