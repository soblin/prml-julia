module beta_dist

using SpecialFunctions

export BetaDist
export pdf

mutable struct BetaDist
    _n_ones::Float64
    _n_zeros::Float64
end

function pdf(beta::BetaDist, mu::Float64)
    a = beta._n_ones;
    b = beta._n_zeros;
    return gamma(a + b) / (gamma(a) * gamma(b)) * mu^(a-1.0) * (1.0 - mu)^(b-1.0)
end

function pdf(beta::BetaDist, mu::Array{Float64, 1})
    a = beta._n_ones;
    b = beta._n_zeros;
    return gamma(a + b) / (gamma(a) * gamma(b)) * mu.^(a-1.0) .* (1.0 .- mu).^(b-1.0)
end

end # module
