@testset "ch2" begin
    Random.seed!(1234)
    rng = MersenneTwister(1234)
    function create_toy_data(func, sample_size, std)
        x = collect(range(0, stop = 1.0, length = sample_size))
        noise = rand(Normal(0.0, std), sample_size)
        return x, func(x) + noise
    end

    function sinusoidal(x)
        return sin.(2 * pi * x)
    end

    @testset "ch2_1" begin
        mu = [0.0, 1.0, 1.0, 1.0]
        bernoulli = BernoulliDist(mu)
        pdf(bernoulli, 0)
    end

    @testset "ch2_2" begin
        x = collect(range(0, stop = 1.0, length = 100))
        for (index, values) in enumerate([[0.1, 0.1], [1, 1], [2, 3], [8, 4]])
            a, b = values[1], values[2]
            beta = BetaDist(a, b)
            y = pdf(beta, x)
        end
    end

    @testset "ch2_3" begin
        x = collect(range(0, stop = 1.0, length = 100))

        beta = BetaDist(2, 2)

        bern = BernoulliDist(beta)
        fitting(bern, [1.0])
    end

    @testset "ch2_4" begin
        println("Maximum likehood estimation")

        model = BernoulliDist([1.0])
        count = draw(model, 10000)
        println("$(count) out of 10000 is label-1")

        println("Beysian estimation")
        model = BernoulliDist(BetaDist(1.0, 1.0))
        fitting(model, [1.0])
        count = draw(model, 10000)
        println("$(count) out of 10000 is label-1")
    end

    @testset "ch2_5" begin
        model = CategoricalDist([0.0])
        fitting(model, [[0.0 1.0 0.0]; [1.0 0.0 0.0]; [0.0 1.0 0.0]])
        println(model._mu)

        mu = DirichletDist(ones(3))
        model = CategoricalDist(mu)
        println("prior")
        println(model._dirichlet._alpha)
        trials = [[1.0 0.0 0.0]; [1.0 0.0 0.0]; [0.0 1.0 0.0]]
        fitting(model, trials)
        println("posterior")
        println(model._dirichlet._alpha)
    end

    @testset "ch2_6" begin
        dirichlet1 = DirichletDist([0.1, 0.1, 0.1])
        dirichlet2 = DirichletDist([1.0, 1.0, 1.0])
        dirichlet3 = DirichletDist([10.0, 10.0, 10.0])
    end

    @testset "ch2_7" begin
        mu_prior = GaussianDist(0.0, 0.1)
        model = GaussianBayesMeanDist(mu_prior, 0.1)

        x = collect((range(-1, stop = 1.0, length = 200)))

        fitting(model, [rand(Normal(0.8, 0.01))])

        fitting(model, [rand(Normal(0.8, 0.01))])
    end

    @testset "ch2_8" begin
        x = collect(range(0, stop = 2.0, length = 100))
        for (index, values) in enumerate([[0.1, 0.1], [1.0, 1.0], [2.0, 3.0], [4.0, 6.0]])
            gamma = GammaDist(values[1], values[2])
        end
    end

    @testset "ch2_9" begin
        X = rand(Uniform(1, 3), 2, 100)
        gaussian = MultivariateGaussianDist(2)
        fitting(gaussian, X)

        x = range(-5, stop = 10, length = 100)
        y = range(-5, stop = 10, length = 100)
        grids = [[i, j] for i in x, j in y]
        vals = [pdf(gaussian, grids[i, j]) for i = 1:100, j = 1:100]
    end
end
