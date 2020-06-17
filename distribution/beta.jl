module beta_dist

export BetaDist
export pdf

mutable struct BetaDist
    _n_ones::Int64
    _n_zeros::Int64
end

function pdf(beta::BetaDist, )
end # module
