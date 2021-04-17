mutable struct GaussianDist
    _mu::Float64
    _var::Float64 # sigma^2
    _precision::Float64 # 1/sigma^2
    function GaussianDist(mu::Float64, var::Float64)
        new(mu, var, 1.0 / var)
    end
end

mutable struct GaussianBayesMeanDist
    _mu_gauss::GaussianDist
    _var::Float64
    _precision::Float64 # 1/sigma^2
    function GaussianBayesMeanDist(mu_gauss::GaussianDist, var::Float64)
        new(mu_gauss, var, 1.0 / var)
    end
end

mutable struct GaussianBayesVarDist
    _var_gamma::GammaDist
end

function pdf(gauss::GaussianDist, x::Float64)
    d = x - gauss._mu
    return exp(-0.5 * d^2 / gauss._var) / sqrt(2 * pi * gauss._var)
end

function pdf(gauss::GaussianDist, X::AbstractArray{Float64,1})
    d = X .- gauss._mu
    return exp.(-0.5 * d .^ 2 / gauss._var) / sqrt(2 * pi * gauss._var)
end

function fitting(gauss::GaussianDist, X::AbstractArray{Float64,1})
    gauss._mu = mean(X)
    gauss._var = var(X)
    gauss._precision = 1.0 / gauss._var
end

function fitting(gauss::GaussianBayesMeanDist, X::AbstractArray{Float64,1})
    # mu and sigma for prior
    mu_0 = gauss._mu_gauss._mu
    precision_0 = gauss._mu_gauss._precision
    precision = gauss._precision # precision of ground-truth distribution

    N = size(X)[1]
    mu_ML = mean(X)

    precision_N = precision_0 + N * precision
    mu_N = (precision_0 / precision_N) * mu_0 + (N * precision / precision_N) * mu_ML

    gauss._mu_gauss._mu = mu_N
    gauss._mu_gauss._var = 1.0 / precision_N
    gauss._mu_gauss._precision = precision_N
end

function fitting(gauss::GaussianBayesVarDist, X::AbstractArray{Float64,1})
    sigma2_ML = var(X)
    N = size(X)[1]

    if N != 1
        gauss._var_gamma._a += N / 2.0
        gauss._var_gamma._b += N / 2.0 * sigma2_ML
    end
end
