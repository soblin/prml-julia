mutable struct CategoricalDist
    _mu::AbstractArray{Float64,1}
    _dirichlet::DirichletDist
    _bayes::Bool
    function CategoricalDist(mu::AbstractArray{Float64,1})
        new(mu, DirichletDist([0.0]), false)
    end
    function CategoricalDist(dirichlet::DirichletDist)
        new([0.0], dirichlet, true)
    end
end

function fitting(dist::CategoricalDist, X::AbstractArray{Float64,1})
    if (dist._bayes == false)
        dist._mu = X
    else
        dist._dirichlet._alpha += X
    end
end

function fitting(dist::CategoricalDist, X::AbstractArray{Float64,2})
    if (dist._bayes == false)
        dist._mu = reshape(mean(X, dims = 1), size(X)[2])
    else
        dist._dirichlet._alpha += reshape(sum(X, dims = 1), size(X)[2])
    end
end
