using LinearAlgebra

mutable struct RidgeRegressor
    _w::AbstractArray{Float64, 1}
    _alpha::Float64
end

function fitting(regressor::RidgeRegressor, Phi::AbstractArray{Float64, 2}, t::AbstractArray{Float64, 1})
    # Phi is the design matrix of size (n_samples, n_feature)
    # t is the target vector of size (sample_size, 1)
    # `pinv(Phi)` = (Phi^T * Phi)^(-1) * Phi^T
    alpha = regressor._alpha;
    n_feature = size(Phi)[2];
    @assert size(Phi)[1] == size(t)[1]
    E = Matrix(I, n_feature, n_feature);
    regressor._w = inv(transpose(Phi) * Phi + alpha * E) * transpose(Phi) * t;
end

function predict(regressor::RidgeRegressor, phi::AbstractArray{Float64, 1})
    # phi is the transformed vector of size n_features
    y = transpose(phi) * regressor._w;
    
    return y
end

function predict(regressor::RidgeRegressor, Phi::AbstractArray{Float64, 2})
    # `Phi` is the transformed feature of x, of size (n_feature, sample_size)
    y = Phi * regressor._w;
    
    return y
end
