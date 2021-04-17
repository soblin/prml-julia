@testset "ch03" begin
    Random.seed!(1234)
    rng = MersenneTwister(1234)

    function create_toy_data(func, sample_size, std, domain = [0.0, 1.0])
        x = collect(range(domain[1], stop = domain[2], length = sample_size))
        shuffle!(rng, x)
        noise = rand(Normal(0.0, std), sample_size)
        return x, func(x) + noise
    end

    function sinusoidal(x)
        return sin.(2 * pi * x)
    end

    @testset "ch03_1" begin
        x = collect(range(-1.0, stop = 1.0, length = 100))
        means = collect(range(-1.0, stop = 1.0, length = 12))
        feature_poly = PolynomialFeature(11)
        feature_gauss = GaussianFeature(means, 0.1)
        feature_sigmoid = SigmoidalFeature(means, 10.0)

        X_polynomial = transform(feature_poly, x)
        X_gaussian = transform(feature_gauss, x)
        X_sigmoidal = transform(feature_sigmoid, x)
    end

    @testset "ch03_2" begin
        x_train, y_train = create_toy_data(sinusoidal, 10, 0.4)
        x_test = collect(range(0, stop = 1.0, length = 100))
        y_test = sinusoidal(x_test)
        means = collect(range(0, stop = 1.0, length = 7))
        feature = GaussianFeature(means, 0.1)

        X_train = transform(feature, x_train)
        X_test = transform(feature, x_test)
        model = LinearRegressor([0], 0)
        fitting(model, X_train, y_train)

        y, var = predict(model, X_test, true)

        model = RidgeRegressor([0.0], 1e-2)
        fitting(model, X_train, y_train)
        y = predict(model, X_test)
    end

    @testset "ch03_3" begin
        means = collect(range(0, stop = 1.0, length = 24))
        feature = GaussianFeature(means, 0.1)

        for a in [1e2, 1.0, 1e-9]
            y_list = []
            for i = 1:100
                x_train, y_train = create_toy_data(sinusoidal, 25, 0.25)
                x_test = collect(range(0, stop = 1.0, length = 100))
                X_train = transform(feature, x_train)
                X_test = transform(feature, x_test)
                model = BayesianRegressor(a, 1.0, 24)
                fitting(model, X_train, y_train)
                y = predict(model, X_test)
            end
        end
    end

    @testset "ch03_4" begin
        function linear(x)
            return -0.3 .+ 0.5 .* x
        end

        x_train, y_train = create_toy_data(linear, 20, 0.1, [-1.0, 1.0])
        x = collect(range(-1.0, stop = 1.0, length = 100))
        y = collect(range(-1.0, stop = 1.0, length = 100))
        #grids = [[i, j] for i in x, j in y];

        model = BayesianRegressor(1.0, 100.0, 2)
        feature = PolynomialFeature(1)
        X_train = transform(feature, x_train)
        X = transform(feature, x)

        dist = MvNormal(model._w_mean, inv(model._w_precision))
        vals = [Distributions.pdf(dist, [i, j]) for j in y, i in x]

        Y_random = predictSampling(model, X, 6)

        for (index, values) in enumerate([[1, 1], [2, 2], [3, 3], [4, 20]])
            first = values[1]
            last = values[2]
            fitting(model, X_train[first:last, :], y_train[first:last])
            dist = MvNormal(model._w_mean, inv(model._w_precision))
            vals = [Distributions.pdf(dist, [i, j]) for j in y, i in x]

            Y_random = predictSampling(model, X, 6)
        end
    end

    @testset "ch03_5" begin
        x_train, y_train = create_toy_data(sinusoidal, 25, 0.25)
        x_test = collect(range(0, stop = 1.0, length = 100))
        y_test = sinusoidal(x_test)

        means = collect(range(0, stop = 1.0, length = 9))
        feature = GaussianFeature(means, 0.1)
        X_train = transform(feature, x_train)
        X_test = transform(feature, x_test)

        model = BayesianRegressor(1e-3, 2.0, 9)

        for (index, ranges) in
            enumerate([[1, 1], [2, 2], [3, 4], [5, 9], [10, 15], [16, 25]])
            first, last = ranges[1], ranges[2]
            fitting(model, X_train[first:last, :], y_train[first:last])
            y, y_std = predict(model, X_test, true)
        end
    end

    @testset "ch03_6" begin
        function cubic(x)
            return x .* (x .- 5.0) .* (x .+ 5.0)
        end

        x_train, y_train = create_toy_data(cubic, 30, 10, [-5.0, 5.0])
        x_test = collect(range(-5.0, stop = 5.0, length = 100))
        evidences = []
        models = []

        for i = 1:6
            feature = PolynomialFeature(i)
            X_train = transform(feature, x_train)
            model = EmpiricalBayesianRegressor(100.0, 100.0, i + 1)
            fitting(model, X_train, y_train, 100)
            push!(evidences, log_evidence(model, X_train, y_train))
            push!(models, model)
        end

        degree = findall(isequal(minimum(evidences)), evidences)
        regressor = models[degree[1]]
        feature = PolynomialFeature(degree[1])

        X_test = transform(feature, x_test)
        y, y_std = predict(regressor, X_test, true)
    end
end
