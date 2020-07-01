include("layer.jl")

mutable struct NeuralNetwork
    layers::Array{AbstractLayer, 1}
    cost_function::AbstractCostFunction
    n_layers::Int64 # cache the number of layers
    function NeuralNetwork(layers_::Array{AbstractLayer, 1}, cost_function_::AbstractCostFunction)
        new(layers, cost_function, size(layers)[1])
    end
end

function fitting(nn::NeuralNetwork, X::Array{Float64, 2}, t::Array{Float64, 2}, learning_rate::Float64)
    # X = [x_1 x_2 ,,, x_N]
    # t = [t_1 t_2 ,,, t_N]
    for layer in nn.layers
        X = forward_propagation(layer, X)
    end

    typeof(X) == typeof(t) || error("fitting: size does not match")

    diff = X - t

    for layer in reverse(nn.layers)
        diff = backward_propagation(layer, diff, learning_rate)
    end

end

function predict(nn::NeuralNetwork, X::Array{Float64, 2})
    # X = [x_1 x_2 ,,, x_N]
    for layer in nn.layers
        X = forward_propagation(layer, X)
    end

    return X
end
