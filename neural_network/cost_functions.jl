abstract type AbstractCostFunction end

struct SigmoidCrossEntropy <: AbstractCostFunction
    name::String
    function SigmoidCrossEntropy(name::String="sigmimd cross entropy")
        new(name)
    end
end

struct SoftmaxCrossEntropy <: AbstractCostFunction
    name::String
    function SoftmaxCrossEntropy(name::String="softmax cross entropy")
        new(name)
    end
end

struct SumSquareError <: AbstractCostFunction
    name::String
    function SumSquareError(name::String="sum square error")
        new(name)
    end
end

function delta(cost_fn::AbstractCostFunction, output::AbstractArray{Float64, 2}, targets::AbstractArray{Float64, 2})
    # output is the output of final layer
    # [output_1 output_2 ,,, output_N]
    # [target_1 target_2 ,,, target_N]
    if typeof(cost_fn) != GaussianMixtureError
        size(output) == size(targets) || error("delta: size does not match")
    end
    return delta_impl(cost_fn, output, targets)
end

function delta_impl(cost_fn::SigmoidCrossEntropy, output::AbstractArray{Float64, 2}, targets::AbstractArray{Float64, 2})
    # `output` denotes "logits"
    probs = 1.0 ./ (1.0 .+ exp.(-output))
    return probs - targets
end

function delta_impl(cost_fn::SoftmaxCrossEntropy, output::AbstractArray{Float64, 2}, targets::AbstractArray{Float64, 2})
    output_max = [maximum(output[:, i]) for i in 1:size(output)[2]]
    output_ = output .- transpose(output_max)
    output_ = exp.(output_)
    probs = output_ ./ sum(output_, dims=1)

    typeof(probs) == typeof(targets) || error("delta_impl: size does not match")
    return probs - targets
end

function delta_impl(cost_fn::SumSquareError, output::AbstractArray{Float64, 2}, targets::AbstractArray{Float64, 2})
    return output - targets
end

function cost(cost_fn::AbstractCostFunction, output::AbstractArray{Float64, 2}, targets::AbstractArray{Float64, 2})
    size(output) == size(targets) || error("cost: size does not match")
    return cost_impl(cost_fn, output, targets)
end

function cost_impl(cost_fn::SigmoidCrossEntropy, output::AbstractArray{Float64, 2}, targets::AbstractArray{Float64, 2})
    probs = 1.0 ./ (1.0 .+ exp.(-output))
    p = clamp.(probs, 1e-10, 1.0 - 1e-10)

    tmp = -( targets .* log.(p) ) - ( (1.0 .- targets) .* log.(1.0 .- p) )
    return sum(tmp)
end

function cost_impl(cost_fn::SoftmaxCrossEntropy, ouput::AbstractArray{Float64, 2}, targets::AbstractArray{Float64, 2})
    output_max = [maximum(output[:, i]) for i in 1:size(output)[2]]
    output_ = output .- transpose(output_max)
    output_ = exp.(output_)
    probs = output_ ./ sum(output_, dims=1)

    p = clamp.(probs, 1e-10, 1.0 - 1e-10)
    return -sum(targets .* log.(p))
end

function cost_impl(cost_fn::SumSquareError, output::AbstractArray{Float64, 2}, targets::AbstractArray{Float64, 2})
    diff = output - targets
    return sum(diff .* diff)
end

# not working yet
struct GaussianMixtureError <: AbstractCostFunction
    name::String
    _n_components::Int64
    function GaussianMixtureError(n_components::Int64, name::String="gaussian mixture error")
        new(name, n_components)
    end
end

function gaussian(x::Float64, mu::Float64, sigma2::Float64)
    return exp(-(x - mu)^2 / (2 * sigma2)) / sqrt(2.0 * pi * sigma2)
end

function delta_impl(cost_fn::GaussianMixtureError, output::AbstractArray{Float64, 2}, targets::AbstractArray{Float64, 2})
    # output is the output of final layer
    # [output_1 output_2 ,,, output_N] of size(n_feature, n_samples)
    # [target_1 target_2 ,,, target_N] of size(1, n_samples)
    n_samples = size(targets)[2]
    n_components = cost_fn._n_components
    @assert size(output)[1] == 3 * n_components

    ret = zeros(3*n_components, n_samples)
    for n in 1:n_samples
        pis = view(output, 1:n_components, n)
        mus = view(output, n_components+1:2*n_components, n)
        sigma2s = view(output, 2*n_components+1:3*n_components, n)

        # (1) convert output to [pis, mus, sigma2s]
        # pis
        max_pi = maximum(pis)
        for k in 1:n_components
            pis[k] = pis[k] - max_pi
        end
        for k in 1:n_components
            pis[k] = exp(pis[k])
        end
        sum_pis = sum(pis)
        for k in 1:n_components
            pis[k] /= sum_pis
        end
        # sigma2s
        for k in 1:n_components
            sigma2s[k] = exp(sigma2s[k])^2
        end

        # (2) compute the graident
        # likelihoods of t_n for each gaussian pi_k N(t_n | mu_k, sigma_k)
        t_n = targets[1, n]
        probs = [pis[k] * gaussian(t_n, mus[k], sigma2s[k]) for k in 1:n_components]
        gammas = probs ./ sum(probs)
        ret[1:n_components, n] = pis - gammas
        ret[n_components+1:2*n_components, n] = gammas .* (mus .- t_n) ./ sigma2s
        ret[2*n_components+1:3*n_components, n] = gammas .* (1.0 .- (t_n .- mus) .* (t_n .- mus) ./ sigma2s)
    end

    return ret
end

function cost_impl(cost_fn::GaussianMixtureError, output::AbstractArray{Float64, 2}, targets::AbstractArray{Float64, 2})
    n_samples = size(targets)[1]
    n_components = cost_fn._n_components
    @assert size(output)[1] == 3 * n_components
    
    cost = 0.0
    for n in 1:n_samples
        pis = view(output, 1:n_components, i)
        mus = view(output, n_components+1:2*n_components, i)
        sigma2s = view(output, 2*n_components+1:3*n_components, i)

        # (1) convert output to [pis, mus, sigma2s]
        # pis
        max_pi = maximum(pis)
        for k in 1:n_components
            pis[k] = pis[k] - max_pi
        end
        for k in 1:n_components
            pis[k] = exp(pis[k])
        end
        sum_pis = sum(pis)
        for k in 1:n_components
            pis[k] /= sum_pis
        end
        # sigma2s
        for k in 1:n_components
            sigma2s[k] = exp(sigma2s[k])
        end

        ln_sum = 0.0
        for k in 1:n_components
            ln_sum += pis[k] * gaussian(targets[n], mus[k], sigma2s[k])
        end
        cost -= log(ln_sum)
    end

    return cost
end

