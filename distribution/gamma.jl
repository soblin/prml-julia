using SpecialFunctions

mutable struct GammaDist
    _a::Float64
    _b::Float64
    function GammaDist(a::Float64, b::Float64)
        new(a, b);
    end
end

function pdf(gamma::GammaDist, x::Float64)
    a = gamma._a;
    b = gamma._b;
    return (b^a) * (x^(a-1)) * exp(-b * x) / gamma(a)
end

function pdf(dist::GammaDist, X::AbstractArray{Float64, 1})
    a = dist._a;
    b = dist._b;
    return (b^a) .* (x.^(a-1)) .* exp.(-b .* x) ./ gamma(a)
end
