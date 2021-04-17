@testset "ch06" begin
    Random.seed!(1234)
    rng = MersenneTwister(1234)
    function create_toy_data(func, n_samples::Int64, std = 1.0, domain = [0.0, 1.0])
        x = collect(range(domain[1], domain[2], length = n_samples))
        shuffle!(rng, x)
        noise = rand(Uniform(-std, std), n_samples)
        return x, func.(x) + noise
    end

    function sinusoidal(x)
        return sin.(2 * pi .* x)
    end

    @testset "ch06_1" begin
        x_train, y_train = create_toy_data(sinusoidal, 10, 0.2)

        x_train = reshape(x_train, 1, size(x_train)[1])
        model = GaussianProcessRegressor(PolynomialKernel(3, 1.0), 1e10)
        fitting(model, x_train, y_train)

        x = reshape(collect(range(0.0, 1.0, length = 100)), 1, 100)
        y, y_sigmas = predict(model, x, true)
    end

    @testset "ch06_2" begin
        x_train, y_train = create_toy_data(sinusoidal, 7, 0.1, [0.0, 0.7])
        x_train = reshape(x_train, 1, size(x_train)[1])
        model = GaussianProcessRegressor(RBFKernel([1.0, 15.0]), 100)
        fitting(model, x_train, y_train)

        x = reshape(collect(range(0.0, 1.0, length = 100)), 1, 100)
        y, y_sigmas = predict(model, x, true)

        y_std = sqrt.(reshape(y_sigmas, length(y_sigmas)))
    end
end
