mutable struct NeuralNetwork
    layers::AbstractArray{AbstractLayer,1}
    cost_function::AbstractCostFunction
    n_layers::Int64 # cache the number of layers
    function NeuralNetwork(
        layers_::AbstractArray{AbstractLayer,1},
        cost_function_::AbstractCostFunction,
    )
        new(copy(layers_), cost_function_, size(layers_)[1])
    end
end

function fitting(
    nn::NeuralNetwork,
    X_::AbstractArray{Float64,2},
    t::AbstractArray{Float64,2},
    learning_rate::Float64,
    policy = "Adam",
)
    # X = [x_1 x_2 ,,, x_N]
    # t = [t_1 t_2 ,,, t_N]
    X = copy(X_)
    for layer in nn.layers
        X = forward_propagation(layer, X)
    end

    typeof(X) == typeof(t) || error("fitting: size does not match")

    diff = X - t

    for layer in reverse(nn.layers)
        diff = backward_propagation(layer, diff, learning_rate, policy)
    end

end

function predict(nn::NeuralNetwork, X_::AbstractArray{Float64,2})
    # X = [x_1 x_2 ,,, x_N]
    X = copy(X_)
    for layer in nn.layers
        X = forward_propagation(layer, X)
    end

    return X
end
