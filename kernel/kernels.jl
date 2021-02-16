abstract type Kernel end

struct PolynomialKernel <: Kernel
    # return (x^T @ y)^M
    _degree::Int64 # degree of polynomial feature
    _intercept::Float64 # intercept
    function PolynomialKernel(degree::Int64, intercept=0.0)
        new(degree, intercept)
    end
end

"""
    kernel(kernel::PolynomialKernel, x::AbstractArray{Float64, 2}, y::AbstractArray{Float64, 2})

For x = [x_1,... x_N] of size (n_feature, n_samples1) and y = [y_1,... y_N] of size (n_feature, n_samples2),
 compute the gram matrix K of size(n_samples1, n_samples2), where K_ij =(x_i * y_j)^M.
"""
function kernel(kernel::PolynomialKernel, x::AbstractArray{Float64, 2}, y::AbstractArray{Float64, 2})
    n_samples1 = size(x)[2]
    n_samples2 = size(y)[2]
    n_feature = size(x)[1]

    intercept = kernel._intercept
    degree = kernel._degree
    
    n_feature == size(y)[1] || error("feature dimension does not match")
    ret = zeros(n_samples1, n_samples2)
    for i in 1:n_samples1
        for  j in 1:n_samples2
            x_i = view(x, :, i)
            y_j = view(y, :, j)
            ret[i, j] = (sum(x_i .* y_j) + intercept)^degree
        end
    end

    return ret
end

"Kernel to compute c0 * exp(-0.5 * c1^T * (x1 - x2)^2) where c0 is scalar and c1 is Array{Float64, 1}"
mutable struct RBFKernel <: Kernel
    _params::Array{Float64, 1}
    _n_params::Int64
    function RBFKernel(params::Array{Float64, 1})
        new(copy(params), size(params)[1])
    end
end

function rbf(c0::Float64, c1::AbstractArray{Float64, 1}, x1::AbstractArray{Float64, 1}, x2::AbstractArray{Float64, 1})
    d = (x1 - x2).^2
    return c0 * exp(-0.5 * transpose(c1) * d)
end

"""
    kernel(kernel::PolynomialKernel, x::AbstractArray{Float64, 2}, y::AbstractArray{Float64, 2})

For x = [x_1,... x_N] of size (n_feature, n_samples1) and y = [y_1,... y_N] of size (n_feature, n_samples2),
 compute the gram matrix K of size(n_samples1, n_samples2), where K_ij = rbf(x_i, x_j).
"""
function kernel(kernel::RBFKernel, x::AbstractArray{Float64, 2}, y::AbstractArray{Float64, 2})
    n_feature = size(x)[1]
    n_samples1 = size(x)[2]
    n_samples2 = size(y)[2]
    n_feature == size(y)[1] || error("size does not match")
    n_feature == kernel._n_params-1 || error("kernel size does not match")
    
    ret = zeros(n_samples1, n_samples2)
    for i in 1:n_samples1
        for j in 1:n_samples2
            x_i = x[:, i]
            y_j = y[:, j]
            ret[i, j] = rbf(kernel._params[1], kernel._params[2:end], x_i, y_j)
        end
    end

    return ret
end

"""
    derivatives(kernel::RBFKernel, x::AbstractArray{Float64, 2}, y::AbstractArray{Float64, 2})

Return the derivative of K(gram matrix) with respect to c0 and c1 as Array{Float64, 3}
"""
function derivatives(kernel::RBFKernel, x::AbstractArray{Float64, 2}, y::AbstractArray{Float64, 2})
    n_feature = size(x)[1]
    n_samples1 = size(x)[2]
    n_samples2 = size(y)[2]
    n_feature == size(y)[1] || error("size does not match")
    n_feature == kernel._n_params-1 || error("kernel size does not match")
    n_params = kernel._n_params
    
    ret = zeros(n_params, n_samples1, n_samples2)
    c0 = kernel._params[1]
    c1 = kernel._params[2:end]
    for i in 1:n_samples1
        for j in 1:n_samples2
            x_i = x[:, i]
            y_j = y[:, j]
            d = (x_i - y_j).^2
            delta_c0 = exp(-0.5 * sum(c0 * d))
            delta_c1 = -0.5 * d * c0 * delta_c0
            ret[:, i, j] = vcat([delta_c0], delta_c1)
        end
    end

    return ret
end

function update_parameters(kernel::RBFKernel, updates_to_add::AbstractArray{Float64, 1})
    kernel._params += updates_to_add
end
