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

function delta(cost_fn::AbstractCostFunction, output::Array{Float64, 2}, targets::Array{Float64, 2})
    # output is the output of final layer
    # [output_1 output_2 ,,, output_N]
    # [target_1 target_2 ,,, target_N]
    size(output) == size(targets) || error("delta: size does not match")
    return delta_impl(cost_fn, output, targets)
end

function delta_impl(cost_fn::SigmoidCrossEntropy, output::Array{Float64, 2}, targets::Array{Float64, 2})
    # `output` denotes "logits"
    probs = 1.0 ./ (1.0 .+ exp.(-output))
    return probs - targets
end

function delta_impl(cost_fn::SoftmaxCrossEntropy, output::Array{Float64, 2}, targets::Array{Float64, 2})
    output_max = [maximum(output[:, i]) for i in 1:size(output)[2]]
    output_ = output .- transpose(output_max)
    output_ = exp.(output_)
    probs = output_ ./ sum(output_, dims=1)

    typeof(probs) == typeof(targets) || error("delta_impl: size does not match")
    return probs - targets
end

function delta_impl(cost_fn::SumSquareError, output::Array{Float64, 2}, targets::Array{Float64, 2})
    return output - targets
end

function cost(cost_fn::AbstractCostFunction, output::Array{Float64, 2}, targets::Array{Float64, 2})
    size(output) == size(targets) || error("cost: size does not match")
    return cost_impl(cost_fn, output, targets)
end

function cost_impl(cost_fn::SigmoidCrossEntropy, output::Array{Float64, 2}, targets::Array{Float64, 2})
    probs = 1.0 ./ (1.0 .+ exp.(-output))
    p = clamp.(probs, 1e-10, 1.0 - 1e-10)

    tmp = -( targets .* log.(p) ) - ( (1.0 .- targets) .* log.(1.0 .- p) )
    return sum(tmp)
end

function cost_impl(cost_fn::SoftmaxCrossEntropy, ouput::Array{Float64, 2}, targets::Array{Float64, 2})
    output_max = [maximum(output[:, i]) for i in 1:size(output)[2]]
    output_ = output .- transpose(output_max)
    output_ = exp.(output_)
    probs = output_ ./ sum(output_, dims=1)

    p = clamp.(probs, 1e-10, 1.0 - 1e-10)
    return -sum(targets .* log.(p))
end

function cost_impl(cost_fn::SumSquareError, output::Array{Float64, 2}, targets::Array{Float64, 2})
    diff = output - targets
    return sum(diff .* diff)
end
