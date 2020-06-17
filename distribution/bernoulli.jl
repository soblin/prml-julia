module bernoulli_dist

export BernoulliDist
export pdf

mutable struct BernoulliDist
    _mu::Float64
    function BernoulliDist(mu::Array{Float64, 1})
        _mu::Float64 = sum(mu) / size(mu)[1];
        new(_mu);
    end
end

function pdf(dist::BernoulliDist, x::Int64)
    return dist._mu^x * (1.0 - dist._mu)^(1 - x)
end

end # module
