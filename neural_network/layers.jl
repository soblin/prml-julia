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
    # for Momentum SGD, RMSProp, and Adam
    W_momentum::Array{Float64, 2}
    W_momentum_aux::Array{Float64, 2}
    b_momentum::Array{Float64, 1}
    b_momentum_aux::Array{Float64, 1}
    param::Float64
    param_aux::Float64
    function LinearLayer(dim_input::Int64, dim_output::Int64, std::Float64=1.0, bias::Float64=0.0, param::Float64=0.9, param_aux::Float64=0.9)
        trunc = truncated(Normal(0.0, std), -2.0*std, 2.0*std);
        W = collect(reshape(rand(trunc, dim_output, dim_input), dim_output, dim_input));
        b = ones(dim_output) * bias;
        new(W, b, zeros(1, 1), zeros(1, 1), dim_input, dim_output, zeros(Float64, size(W)), zeros(Float64, size(W)), zeros(Float64, size(b)), zeros(Float64, size(b)), param, param_aux);
    end
end

mutable struct SigmoidLayer <: AbstractLayer
    W::Array{Float64, 2}
    b::Array{Float64, 1}
    input::Array{Float64, 2}
    output::Array{Float64, 2}
    dim_input::Int64
    dim_output::Int64
    # for Momentum SGD, RMSProp, and Adam
    W_momentum::Array{Float64, 2}
    W_momentum_aux::Array{Float64, 2}
    b_momentum::Array{Float64, 1}
    b_momentum_aux::Array{Float64, 1}
    param::Float64
    param_aux::Float64
    function SigmoidLayer(dim_input::Int64, dim_output::Int64, std::Float64=1.0, bias::Float64=0.0, param::Float64=0.9, param_aux::Float64=0.9)
        trunc = truncated(Normal(0.0, std), -2.0*std, 2.0*std);
        W = collect(reshape(rand(trunc, dim_output, dim_input), dim_output, dim_input));
        b = ones(dim_output) * bias;
        new(W, b, zeros(1, 1), zeros(1, 1), dim_input, dim_output, zeros(Float64, size(W)), zeros(Float64, size(W)), zeros(Float64, size(b)), zeros(Float64, size(b)), param, param_aux);
    end
end

mutable struct TanhLayer <: AbstractLayer
    W::Array{Float64, 2}
    b::Array{Float64, 1}
    input::Array{Float64, 2}
    output::Array{Float64, 2}
    dim_input::Int64
    dim_output::Int64
    # for Momentum SGD, RMSProp, and Adam
    W_momentum::Array{Float64, 2}
    W_momentum_aux::Array{Float64, 2}
    b_momentum::Array{Float64, 1}
    b_momentum_aux::Array{Float64, 1}
    param::Float64
    param_aux::Float64
    function TanhLayer(dim_input::Int64, dim_output::Int64, std::Float64=1.0, bias::Float64=0.0, param::Float64=0.9, param_aux::Float64=0.9)
        trunc = truncated(Normal(0.0, std), -2.0*std, 2.0*std);
        W = collect(reshape(rand(trunc, dim_output, dim_input), dim_output, dim_input));
        b = ones(dim_output) * bias;
        new(W, b, zeros(1, 1), zeros(1, 1), dim_input, dim_output, zeros(Float64, size(W)), zeros(Float64, size(W)), zeros(Float64, size(b)), zeros(Float64, size(b)), param, param_aux);
    end
end

mutable struct ReLULayer <: AbstractLayer
    W::Array{Float64, 2}
    b::Array{Float64, 1}
    input::Array{Float64, 2}
    output::Array{Float64, 2}
    dim_input::Int64
    dim_output::Int64
    # for Momentum SGD, RMSProp, and Adam
    W_momentum::Array{Float64, 2}
    W_momentum_aux::Array{Float64, 2}
    b_momentum::Array{Float64, 1}
    b_momentum_aux::Array{Float64, 1}
    param::Float64
    param_aux::Float64
    function ReLULayer(dim_input::Int64, dim_output::Int64, std::Float64=1.0, bias::Float64=0.0, param::Float64=0.9, param_aux::Float64=0.9)
        trunc = truncated(Normal(0.0, std), -2.0*std, 2.0*std)
        W = collect(reshape(rand(trunc, dim_output, dim_input), dim_output, dim_input));
        b = ones(dim_output) * bias;
        new(W, b, zeros(1, 1), zeros(1, 1), dim_input, dim_output, zeros(Float64, size(W)), zeros(Float64, size(W)), zeros(Float64, size(b)), zeros(Float64, size(b)), param, param_aux);
    end
end

function forward_propagation(layer::AbstractLayer, x_::Array{Float64, 1})
    @assert layer.dim_input == size(x_)[1]
    x = collect(reshape(x_, size(x_)[1], 1))
    return forward_propagation_impl(layer, x)
end

function forward_propagation(layer::AbstractLayer, X::Array{Float64, 2})
    @assert layer.dim_input == size(X)[1]
    return forward_propagation_impl(layer, X)
end

function forward_propagation_impl(layer::LinearLayer, X::Array{Float64, 2})
    # X is the array of [x1 x2 ... x_N], x1 is a vector of feature_size
    layer.input = copy(X)
    return (layer.W) * X .+ (layer.b)
end

function forward_propagation_impl(layer::SigmoidLayer, X::Array{Float64, 2})
    # X is the array of [x1 x2 ... x_N], x1 is a vector of feature_size
    layer.input = copy(X)
    activation = (layer.W) * X .+ (layer.b)
    layer.output = 1.0 ./ (1.0 .+ exp.(-activation))
    return layer.output
end

function forward_propagation_impl(layer::TanhLayer, X::Array{Float64, 2})
    # X is the array of [x1 x2 ... x_N], x1 is a vector of feature_size
    layer.input = copy(X)
    activation = (layer.W) * X .+ (layer.b)
    layer.output = tanh.(activation)
    return layer.output
end

function forward_propagation_impl(layer::ReLULayer, X::Array{Float64, 2})
    # X is the array of [x1 x2 ... x_N], x1 is a vector of feature_size
    layer.input = copy(X)
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

function backward_propagation(layer::AbstractLayer, delta_output::Array{Float64, 2}, learning_rate::Float64, policy::String="Adam")
    @assert size(delta_output)[1] == layer.dim_output
    if policy == "SGD"
        return backward_propagation_impl_sgd(layer, delta_output, learning_rate)
    elseif policy == "MomentumSGD"
        return backward_propagation_impl_msgd(layer, delta_output, learning_rate)
    elseif policy == "RMSProp"
        return backward_propagation_impl_rmsprop(layer, delta_output, learing_rate)
    elseif policy == "Adam"
        return backward_propagation_impl_adam(layer, delta_output, learning_rate)
    else
        error("Policy must be (SGD|MomentumSGD|RMSProp|Adam)")
    end
end

function backward_propagation_impl_sgd(layer::AbstractLayer, delta_output::Array{Float64, 2}, learning_rate::Float64)
    delta = delta_output .* activate_derivative_impl(layer)
    W = copy(layer.W)
    layer.W -= learning_rate .* (delta * transpose(layer.input))
    layer.b -= learning_rate .* reshape(sum(delta, dims=2), layer.dim_output)

    return transpose(W) * delta
end

function backward_propagation_impl_msgd(layer::AbstractLayer, delta_output::Array{Float64, 2}, learning_rate::Float64)
    delta = delta_output .* activate_derivative_impl(layer)
    W = copy(layer.W)
    W_grad = (delta * transpose(layer.input))
    b_grad = reshape(sum(delta, dims=2), layer.dim_output)

    mu = layer.param
    W_momentum = mu * layer.W_momentum - (1.0 - mu) * learning_rate * W_grad
    b_momentum = mu * layer.b_momentum - (1.0 - mu) * learning_rate * b_grad

    layer.W_momentum = W_momentum
    layer.b_momentum = b_momentum

    layer.W += W_momentum
    layer.b += b_momentum
    
    return transpose(W) * delta
end

function backward_propagation_impl_rmsprop(layer::AbstractLayer, delta_output::Array{Float64, 2}, learning_rate::Float64)
    delta = delta_output .* activate_derivative_impl(layer)
    W = copy(layer.W)
    W_grad = (delta * transpose(layer.input))
    b_grad = reshape(sum(delta, dims=2), layer.dim_output)

    rho = layer.param
    W_momentum = rho * layer.W_momentum + (1.0 - rho) * (W_grad.^2)
    b_momentum = rho * layer.b_momentum + (1.0 - rho) * (b_grad.^2)
    
    layer.W_momentum = W_momentum
    layer.b_momentum = b_momentum

    layer.W -= learning_rate * W_grad ./ sqrt.(W_momentum .+ 1e-8)
    layer.b -= learning_rate * b_grad ./ sqrt.(b_momentum .+ 1e-8)
    
    return transpose(W) * delta
end

function backward_propagation_impl_adam(layer::AbstractLayer, delta_output::Array{Float64, 2}, learning_rate::Float64)
    delta = delta_output .* activate_derivative_impl(layer)
    W = copy(layer.W)
    W_grad = (delta * transpose(layer.input))
    b_grad = reshape(sum(delta, dims=2), layer.dim_output)

    rho1 = layer.param
    rho2 = layer.param_aux

    W_momentum = rho1 * layer.W_momentum + (1.0 - rho1) * W_grad # m
    W_momentum_aux = rho2 * layer.W_momentum_aux + (1.0 - rho2) * (W_grad).^2 # v
    W_momentum_hat = W_momentum / (1.0 - rho1) # \hat{m}
    W_momentum_aux_hat = W_momentum_aux / (1.0 - rho2) # \hat{v}
    b_momentum = rho1 * layer.b_momentum + (1.0 - rho1) * b_grad # m
    b_momentum_aux = rho2 * layer.b_momentum_aux + (1.0 - rho2) * (b_grad).^2 # v
    b_momentum_hat = b_momentum / (1.0 - rho1) # \hat{m}
    b_momentum_aux_hat = b_momentum_aux / (1.0 - rho2) # \hat{v}

    layer.W_momentum = W_momentum
    layer.W_momentum_aux = W_momentum_aux
    layer.b_momentum = b_momentum
    layer.b_momentum_aux = b_momentum_aux

    layer.W -= learning_rate * W_momentum_hat ./ sqrt.(W_momentum_aux_hat .+ 1e-8)
    layer.b -= learning_rate * b_momentum_hat ./ sqrt.(b_momentum_aux_hat .+ 1e-8)
    
    return transpose(W) * delta
end
