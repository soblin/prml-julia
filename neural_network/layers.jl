using LinearAlgebra
using Distributions
abstract type AbstractLayer end

mutable struct LinearLayer <: AbstractLayer
    W::Array{Float64, 2}
    b::Array{Float64, 1}
    input::Array{Float64, 2}
    output::Array{Float64, 2}
    dim_input::Int64
    dim_output::Int64
    function LinearLayer(dim_input::Int64, dim_output::Int64, std::Float64=1.0, bias::Float64=0.0)
        trunc = truncated(Normal(0.0, std), -2.0*std, 2.0*std);
        W = collect(reshape(rand(trunc, dim_output, dim_input), dim_output, dim_input));
        b = ones(dim_output) * bias;
        new(W, b, zeros(1, 1), zeros(1, 1), dim_input, dim_output);
    end
end

mutable struct SigmoidLayer <: AbstractLayer
    W::Array{Float64, 2}
    b::Array{Float64, 1}
    input::Array{Float64, 2}
    output::Array{Float64, 2}
    dim_input::Int64
    dim_output::Int64
    function SigmoidLayer(dim_input::Int64, dim_output::Int64, std::Float64=1.0, bias::Float64=0.0)
        trunc = truncated(Normal(0.0, std), -2.0*std, 2.0*std);
        W = collect(reshape(rand(trunc, dim_output, dim_input), dim_output, dim_input));
        b = ones(dim_output) * bias;
        new(W, b, zeros(1, 1), zeros(1, 1), dim_input, dim_output);
    end
end

mutable struct TanhLayer <: AbstractLayer
    W::Array{Float64, 2}
    b::Array{Float64, 1}
    input::Array{Float64, 2}
    output::Array{Float64, 2}
    dim_input::Int64
    dim_output::Int64
    function TanhLayer(dim_input::Int64, dim_output::Int64, std::Float64=1.0, bias::Float64=0.0)
        trunc = truncated(Normal(0.0, std), -2.0*std, 2.0*std);
        W = collect(reshape(rand(trunc, dim_output, dim_input), dim_output, dim_input));
        b = ones(dim_output) * bias;
        new(W, b, zeros(1, 1), zeros(1, 1), dim_input, dim_output);
    end
end

mutable struct ReLULayer <: AbstractLayer
    W::Array{Float64, 2}
    b::Array{Float64, 1}
    input::Array{Float64, 2}
    output::Array{Float64, 2}
    dim_input::Int64
    dim_output::Int64
    function ReLULayer(dim_input::Int64, dim_output::Int64, std::Float64=1.0, bias::Float64=0.0)
        trunc = truncated(Normal(0.0, std), -2.0*std, 2.0*std)
        W = collect(reshape(rand(trunc, dim_output, dim_input), dim_output, dim_input));
        b = ones(dim_output) * bias;
        new(W, b, zeros(1, 1), zeros(1, 1), dim_input, dim_output);
    end
end

function forward_propagation(layer::AbstractLayer, x::Array{Float64, 1})
    @assert layer.dim_input == size(x)[1]
    x_ = collect(reshape(x, size(x)[1], 1))
    return forward_propagation_impl(layer, x_)
end

function forward_propagation(layer::AbstractLayer, X::Array{Float64, 2})
    @assert layer.dim_input == size(X)[1]
    return forward_propagation_impl(layer, X)
end

function forward_propagation_impl(layer::LinearLayer, X::Array{Float64, 2})
    # X is the array of [x1 x2 ... x_N], x1 is a vector of feature_size
    layer.input = X
    return (layer.W) * X .+ (layer.b)
end

function forward_propagation_impl(layer::SigmoidLayer, X::Array{Float64, 2})
    # X is the array of [x1 x2 ... x_N], x1 is a vector of feature_size
    layer.input = X
    activation = (layer.W) * X .+ (layer.b)
    layer.output = 1.0 ./ (1.0 .+ exp.(-activation))
    return layer.output
end

function forward_propagation_impl(layer::TanhLayer, X::Array{Float64, 2})
    # X is the array of [x1 x2 ... x_N], x1 is a vector of feature_size
    layer.input = X
    activation = (layer.W) * X .+ (layer.b)
    layer.output = tanh.(activation)
    return layer.output
end

function forward_propagation_impl(layer::ReLULayer, X::Array{Float64, 2})
    # X is the array of [x1 x2 ... x_N], x1 is a vector of feature_size
    layer.input = X
    activation = (layer.W) * X .+ (layer.b)
    # layer.output = activation.clip(min=0)
    layer.output = (activation + broadcast(abs, activation)) / 2.0
    return layer.output
end

function activate_derivative(layer::AbstractLayer)
    return activate_derivative_impl(layer)
end

function activate_derivative_impl(layer::LinearLayer)
    # [f'(z_1), f'(z_2),,,, f'(z_N)]
    return ones(layer.dim_output, size(layer.output)[2]) * 1.0
end

function activate_derivative_impl(layer::SigmoidLayer)
    # [f'(z_1), f'(z_2),,,, f'(z_N)]
    return (layer.output) .* (1.0 .- layer.output)
end

function activate_derivative_impl(layer::TanhLayer)
    # [f'(z_1), f'(z_2),,,, f'(z_N)]
    return 1.0 .- (layer.output) .* (layer.output)
end

function activate_derivative_impl(layer::ReLULayer)
    # [f'(z_1), f'(z_2),,,, f'(z_N)]
    return (layer.output .> 0.0) * 1.0
end

function backward_propagation(layer::AbstractLayer, delta_output::Array{Float64, 2}, learning_rate::Float64)
    @assert size(delta_output)[1] == layer.dim_output
    return backward_propagation_impl(layer, delta_output, learning_rate)
end

function backward_propagation_impl(layer::AbstractLayer, delta_output::Array{Float64, 2}, learning_rate::Float64)
    delta = delta_output .* activate_derivative_impl(layer)
    W = copy(layer.W)
    layer.W -= learning_rate .* (delta * transpose(layer.input))
    layer.b -= learning_rate .* reshape(sum(delta, dims=2), layer.dim_output)

    return transpose(W) * delta
end
