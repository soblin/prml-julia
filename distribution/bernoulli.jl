include("beta.jl")

mutable struct BernoulliDist
    _mu::Float64
    _beta::BetaDist
    # if use BetaDist, _bayes == true
    _bayes::Bool
    function BernoulliDist(mu::Array{Float64, 1})
        _mu::Float64 = sum(mu) / size(mu)[1];
        new(_mu, BetaDist(0, 0), false);
    end
    function BernoulliDist(beta::BetaDist)
        new(0, beta, true);
    end
end

function pdf(dist::BernoulliDist, x::Int64)
    return dist._mu^x * (1.0 - dist._mu)^(1 - x)
end

function fitting(dist::BernoulliDist, X::Array{Float64, 1})
    if dist._bayes == false
        return fitting_ml(dist, X);
    else
        return fitting_map(dist, X);
    end
end

function fitting_ml(dist::BernoulliDist, X::Array{Float64, 1})
    # X is the array of 0 or 1
    # X .== 0 is the array indicating if X[i] == 0
    n_zeros = sum(X .== 0.0);
    # X .== 1 is the array indicating if X[i] == 1
    n_ones = sum(X .== 1.0);
    dist._mu = n_ones * 1.0 / (n_ones + n_zeros);
end

function fitting_map(dist::BernoulliDist, X::Array{Float64, 1})
    # X is the array of 0 or 1
    # X .== 0 is the array indicating if X[i] == 0
    n_zeros = sum(X .== 0.0);
    # X .== 1 is the array indicating if X[i] == 1
    n_ones = sum(X .== 1.0);
    dist._beta._n_ones += n_ones;
    dist._beta._n_zeros += n_zeros;
end
