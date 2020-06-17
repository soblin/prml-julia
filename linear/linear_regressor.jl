using LinearAlgebra

mutable struct LinearRegressor
    _w::Array{Float64, 1}
    _var::Float64
end

function fitting(regressor::LinearRegressor, Phi::Array{Float64, 2}, t::Array{Float64, 1})
    # Phi is the design matrix of size (sample_size, n_feature)
    # t is the target vector of size (sample_size)
    # `pinv(Phi)` = (Phi^T * Phi)^(-1) * Phi^T
    w = pinv(Phi) * t
    error = Phi * w - t
    var = sum(error.^2)  / size(t)[1]
    
    regressor._w = w
    regressor._var = var
end

function predict(regressor::LinearRegressor, phi, return_std::Bool)
    # `phi` is the transformed feature of x, of size (sample_size, n_feature)
    y = phi * regressor._w
    std = zeros(1, size(y)[1]) .+ regressor._var
    if return_std == true
        return y, std
    else
        return y
    end
end
