mutable struct DirichletDist
    #=
    Dirichlet Distribution
    p(mu | alpah)
    = gamma(sum(alpha))
    * Prod_k[ mu_k^(alpha_k-1)]
    / gamma(alpha_1) / ... / gamma(alpha_k)
    =#
    _alpha::Array{Float64, 1}
end

