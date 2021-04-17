@testset "ch1" begin
    function create_toy_data(func, sample_size, std)
        x = collect(range(0, stop = 1.0, length = sample_size))
        noise = rand(Normal(0.0, std), sample_size)
        return x, func(x) + noise
    end

    function sinusoidal(x)
        return sin.(2 * pi * x)
    end

    function rmse(a::Array{Float64,1}, b::Array{Float64,1})
        return sum((b - a) .^ 2) / size(a)[1]
    end

    x_train, y_train = create_toy_data(sinusoidal, 10, 0.25)
    x_test = collect(range(0, stop = 1.0, length = 100))
    y_test = sinusoidal(x_test)

    @testset "ch1_1" begin
        feature = PolynomialFeature(9)
        X_train = transform(feature, x_train)
        X_test = transform(feature, x_test)
        #x = collect(reshape(range(0, stop=1.0, length=10), 10));
        model = LinearRegressor([0], 0)
        #Phi = collect(reshape(transpose(X_train), size(X_train)[2], size(X_train)[1]));
        fitting(model, X_train, y_train)
        #tmp = collect(reshape(transpose(X_test), size(X_test)[2], size(X_test)[1]));
        y, y_std = predict(model, X_test, true)
    end

    @testset "ch1_2" begin

        training_errors = []
        test_errors = []

        for i = 0:10
            feature = PolynomialFeature(i)
            X_train = transform(feature, x_train)
            X_test = transform(feature, x_test)

            model = LinearRegressor([0], 0)
            fitting(model, X_train, y_train)
            y_trained = predict(model, X_train, false)
            push!(training_errors, rmse(predict(model, X_train, false), y_train))
            push!(
                test_errors,
                rmse(
                    predict(model, X_test, false),
                    y_test + rand(Normal(0.0, 0.25), size(y_test)[1]),
                ),
            )
        end
    end
end
