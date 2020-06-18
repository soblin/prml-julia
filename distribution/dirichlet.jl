using SpecialFunctions

mutable struct DirichletDist
    #=
    Dirichlet Distribution
    p(mu | alpha)
    = gamma(sum(alpha))
    * Prod_k[ mu_k^(alpha_k-1)]
    / gamma(alpha_1) / ... / gamma(alpha_k)
    =#
    _alpha::Array{Float64, 1}
end

function pdf(dirichlet::DirichletDist, mu::Array{Float64, 1})
    N = size(mu)[1];
    @assert N == size(dirichlet._alpha)[1]
    
    sum_alpha = sum(dirichlet._alpha);
    prod_k = 1.0;
    gamma_k = 1.0;
    for k in 1:N
        alpha_k = dirichlet._alpha[k];
        prod_k *= (mu[k]^(alpha_k-1.0));
        gamma_k *= gamma(alpha_k);
    end
    return gamma(sum_alpha) * prod_k / gamma_k
end

function pdf(dirichlet::DirichletDist, mu::Array{Float64, 2})
    N = size(mu)[1];
    n_samples = size(mu)[2];
    @assert N == size(dirichlet._alpha)[1]

    return [pdf(dirichlet, mu[:, i]) for i in 1:n_samples]
end
