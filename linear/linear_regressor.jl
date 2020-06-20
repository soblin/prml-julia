using LinearAlgebra

mutable struct LinearRegressor
    _w::Array{Float64, 1}
    _var::Float64
end

function fitting(regressor::LinearRegressor, Phi::Array{Float64, 2}, t::Array{Float64, 1})
    # Phi is the transpose of design matrix of size (n_feature, sample_size)
    # t is the target vector of size (sample_size, 1)
    # `pinv(Phi)` = (Phi^T * Phi)^(-1) * Phi^T
    w = pinv(transpose(Phi)) * t
    error = transpose(Phi) * w - t
    var = sum(error.^2)  / size(t)[1]
    
    regressor._w = w
    regressor._var = var
end

function predict(regressor::LinearRegressor, Phi::Array{Float64, 2}, return_std::Bool)
    # `phi` is the transformed feature of x, of size (n_feature, sample_size)
    y = transpose(Phi) * regressor._w;
    std = zeros(1, size(y)[1]) .+ regressor._var
    if return_std == true
        return y, std
    else
        return y
    end
end
