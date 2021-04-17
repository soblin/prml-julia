mutable struct MultivariateGaussianDist
    _mu::AbstractArray{Float64,1}
    _cov::AbstractArray{Float64,2}
    function MultivariateGaussianDist(dim::Int64)
        new(zeros(dim), Matrix{Float64}(I, dim, dim))
    end
end

function pdf(gaussian::MultivariateGaussianDist, x::AbstractArray{Float64,1})
    @assert size(x)[1] == size(gaussian._mu)[1]
    N = size(x)[1]
    d = x - gaussian._mu
    quadratics = transpose(d) * gaussian._cov * d
    @assert typeof(quadratics) == Float64
    return exp(-0.5 * quadratics) / sqrt(det(gaussian._cov)) / (2 * pi)^(0.5 * N)
end

function pdf(gaussian::MultivariateGaussianDist, X::AbstractArray{Float64,2})
    # X consists of list of samples [x_1, x_2, ,,, x_(n_samples)], and x_i is a vertical vector
    @assert size(X)[1] == size(gaussian._mu)[1]
    N = size(X)[1]
    n_samples = size(X)[2]
    # In Julia lang, 1-D AbstractArray is vertical vector
    d = X .- gaussian._mu
    #quadratics = diag(transpose(d) * gaussian._cov * d);
    quadratics = [transpose(d[:, i]) * gaussian._cov * d[:, i] for i = 1:n_samples]
    @assert ndims(quadratics) == 1 && size(quadratics)[1] == n_samples
    return exp.(-0.5 * quadratics) / sqrt(det(gaussian._cov)) / (2 * pi)^(0.5 * N)
end

function fitting(gaussian::MultivariateGaussianDist, X::AbstractArray{Float64,1})
    gaussian._mu = X
    gaussian._cov = cov(X, corrected = true)
end

function fitting(gaussian::MultivariateGaussianDist, X::AbstractArray{Float64,2})
    # X consists of the list of samples [x_1, x_2, ,,, x_(n_samples)]
    @assert size(X)[1] == size(gaussian._mu)[1]
    n_samples = size(X)[2]
    N = size(X)[1]
    gaussian._mu = collect(reshape(mean(X, dims = 2), N))
    gaussian._cov = cov(X, dims = 2, corrected = true)
end
