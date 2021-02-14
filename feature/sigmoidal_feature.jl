mutable struct SigmoidalFeature
    _means::AbstractArray{Float64, 1}
    _coeff::Float64
    _n_kernels::Int64 # cache the number of kernels
    function SigmoidalFeature(means::AbstractArray{Float64, 1}, coeffs::Float64)
        new(means, coeffs, size(means)[1]);
    end
end

function _sigmoid(mean::Float64, coeff::Float64, x::AbstractArray{Float64, 1})
    return tanh.((x .- mean) .* (coeff * 0.5)) .* 0.5 .+ 0.5
end

function transform(feature::SigmoidalFeature, x::AbstractArray{Float64, 1})
    # x is an array of size n_samples
    # returns Phi = [phi_1(x),,,, phi_N(x)] where phi_i(x) is a sigmoidal centered around means[i]
    n_samples = size(x)[1];
    n_kernels = feature._n_kernels;
    Phi = zeros(n_kernels, n_samples);
    for i in 1:n_kernels
        mean = feature._means[i];
        Phi[i, :] = _sigmoid(mean, feature._coeff, x);
    end

    return Phi;
end
