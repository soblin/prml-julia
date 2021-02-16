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
