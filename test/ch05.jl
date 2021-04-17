@testset "ch05" begin
    @testset "ch05_1" begin
        Random.seed!(1234)
        rng = MersenneTwister(1234)

        function create_toy_data(func, sample_size, domain = [0.0, 1.0], noise = 0.1)
            x = collect(range(domain[1], stop = domain[2], length = sample_size))
            shuffle!(rng, x)
            noise = rand(Uniform(-noise, noise), sample_size)
            return x, func.(x) + noise
        end

        function func(x)
            2.0 * x + 0.7 * sin(2 * pi * x)
        end

        function square(x)
            return x^2
        end

        function sinusoidal(x)
            return sin(pi * x)
        end

        function absolute(x)
            if x >= 0.0
                return x
            else
                return -x
            end
        end

        function heaviside(x)
            sign_x = x / absolute(x)
            return 0.5 * (sign_x + 1.0)
        end

        layers = [TanhLayer(1, 4), TanhLayer(4, 3), LinearLayer(3, 1)]
        cost_function = SumSquareError()
        nn = NeuralNetwork(layers, cost_function)

        toy_functions = [square, sinusoidal, absolute, heaviside]
        # "RMSProp" seems to be contains sqrt(-) error
        policies = ["Adam", "SGD", "MomentumSGD", "MomentumSGD"]
        domain = [-1.0, 1.0]

        for (toy_func, policy) in zip(toy_functions, policies)
            x_train, y_train = create_toy_data(toy_func, 100, domain, 0.01)
            X = collect(reshape(x_train, 1, 100))
            t = collect(reshape(y_train, 1, 100))

            for i = 1:10
                X = collect(reshape(x_train, 1, 100))
                t = collect(reshape(y_train, 1, 100))

                fitting(nn, X, t, 0.001, policy)
            end

            x_test = collect(reshape(range(domain[1], domain[2], length = 100), 1, 100))
            X_test = copy(x_test)
            X_test = predict(nn, X_test)

            x_test = collect(reshape(x_test, 100))
            X_test = collect(reshape(X_test, 100))
        end
    end

    @testset "ch5_2" begin
        n_samples = 100
        points = rand(2, n_samples)
        points = points .* 2 .- 1.0

        function boundary(x)
            return 1.2 .* x .* x
        end

        targets = [boundary(points[1, i]) < points[2, i] for i = 1:n_samples] * 1.0

        x_test = collect(range(-1.0, stop = 1.0, length = 100))

        ## Classification
        layers = [TanhLayer(2, 2), SigmoidLayer(2, 1)]
        cost_function = SigmoidCrossEntropy()
        targets = collect(reshape(targets, 1, n_samples))
        nn = NeuralNetwork(layers, cost_function)

        ### Training
        n_training = 5000
        for i = 1:n_training
            fitting(nn, points, targets, 0.001)
        end

        ### Contour of nn prediction
        x = collect(range(-1.0, stop = 1.0, length = 10))
        y = collect(range(-1.0, stop = 1.0, length = 10))
        val = [predict(nn, reshape([i, j], 2, 1))[1] for j in y, i in x]
    end
end
