using LinearAlgebra

mutable struct RidgeRegressor
    _w::Array{Float64, 1}
    _alpha::Float64
end

function fitting(regressor::RidgeRegressor, Phi::Array{Float64, 2}, t::Array{Float64, 1})
    # Phi is the transpose of design matrix of size (n_feature, sample_size)
    # t is the target vector of size (sample_size, 1)
    # `pinv(Phi)` = (Phi^T * Phi)^(-1) * Phi^T
    alpha = regressor._alpha;
    n_feature = size(Phi)[1];
    E = Matrix(I, n_feature, n_feature);
    regressor._w = inv(Phi * transpose(Phi) + alpha * E) * Phi * t;
end

function predict(regressor::RidgeRegressor, phi::Array{Float64, 1})
    # phi is the transformed vector of size n_features
    y = transpose(phi) * regressor._w;
    
    return y
end

function predict(regressor::RidgeRegressor, Phi::Array{Float64, 2})
    # `Phi` is the transformed feature of x, of size (n_feature, sample_size)
    y = transpose(Phi) * regressor._w;
    
    return y
end
