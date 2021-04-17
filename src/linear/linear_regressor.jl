mutable struct LinearRegressor
    _w::AbstractArray{Float64, 1}
    _var::Float64
end

function fitting(regressor::LinearRegressor, Phi::AbstractArray{Float64, 2}, t::AbstractArray{Float64, 1})
    # Phi is the design matrix of size (n_samples, n_feature)
    # t is the target vector of size (sample_size, 1)
    # `pinv(Phi)` = (Phi^T * Phi)^(-1) * Phi^T
    w = pinv(Phi) * t
    error = Phi * w - t
    var = sum(error.^2)  / size(t)[1]
    
    regressor._w = w
    regressor._var = var
end

function predict(regressor::LinearRegressor, phi::AbstractArray{Float64, 1}, return_std::Bool)
    # phi is the transformed vector of size n_features
    y = transpose(phi) * regressor._w;
    std = regressor._var;
    
    if return_std == true
        return y, std
    else
        return y
    end
end

function predict(regressor::LinearRegressor, Phi::AbstractArray{Float64, 2}, return_std::Bool)
    # Phi is the design matrix [phi(x_1), phi(x_2), ,,, phi(x_N)]^T of size (n_samples, n_features)
    y = Phi * regressor._w;
    std = regressor._var;
    
    if return_std == true
        return y, std
    else
        return y
    end
end
