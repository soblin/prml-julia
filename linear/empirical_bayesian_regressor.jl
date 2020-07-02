mutable struct EmpiricalBayesianRegressor
    _alpha::Float64
    _beta::Float64
    _w_mean::Array{Float64, 1}
    _w_precision::Array{Float64, 2}
    _n_features::Int64
    function EmpiricalBayesianRegressor(alpha::Float64, beta::Float64, n_features)
        new(alpha, beta, zeros(n_features) * 1.0, zeros(n_features, n_features) * 1.0, n_features)
    end
end

function fitting(regressor::EmpiricalBayesianRegressor, PhiT::Array{Float64, 2}, t::Array{Float64, 1}, max_iter::Int64=100)
    # PhiT is the array [phi(x_1), phi(x_2), ,,, phi(x_N)]
    n_features = size(PhiT)[1];
    n_samples = size(PhiT)[2];
    @assert n_samples == size(t)[1]

    alpha = regressor._alpha;
    beta = regressor._beta;
    M = PhiT * transpose(PhiT);
    eigenvalues = eigvals(M);
    @assert n_features == size(eigenvalues)[1]

    E = Matrix(I, n_features, n_features);
    w_mean = zeros(n_features);
    w_precision = zeros(n_features, n_features);
    for i in 1:max_iter
        params = [alpha beta];
        w_precision = alpha * E + beta * M;
        w_mean = beta * inv(w_precision) * PhiT * t;
        gamma = sum(eigenvalues ./ (eigenvalues .+ alpha));

        alpha = gamma / norm(w_mean)^2;
        beta = (n_samples - gamma) / sum((transpose(PhiT) * w_mean - t).^2);
        if isapprox(params, [alpha beta])
            break
        end
    end

    # update
    regressor._alpha = alpha;
    regressor._beta = beta;
    regressor._w_mean = w_mean;
    regressor._w_precision = w_precision;
end

function log_evidence(regressor::EmpiricalBayesianRegressor, PhiT::Array{Float64, 2}, t::Array{Float64, 1})
    # PhiT is the array [phi(x_1), phi(x_2), ,,, phi(x_N)]
    n_features = size(PhiT)[1];
    n_samples = size(PhiT)[2];
    M = n_features;
    N = n_samples;
    @assert n_samples == size(t)[1]

    alpha = regressor._alpha;
    beta = regressor._beta;
    E = Matrix(I, n_features, n_features);
    A = alpha * E + beta * PhiT * transpose(PhiT);
    w = regressor._w_mean;
    Ew = beta / 2.0 * sum((t - transpose(PhiT) * w).^2) + alpha / 2.0 * (transpose(w) * w);

    return M / 2.0 * log(alpha) + N / 2.0 * log(beta) - Ew - log(det(A)) / 2.0 - N / 2.0 * log(2 * pi);
end

function predict(regressor::EmpiricalBayesianRegressor, PhiT::Array{Float64, 2}, return_std::Bool=true)
    # PhiT is the array [phi(x_1), phi(x_2), ,,, phi(x_N)]
    N = size(PhiT)[2];
    w_mean = regressor._w_mean;
    w_cov = inv(regressor._w_precision);
    
    y = transpose(PhiT) * w_mean;
    
    if return_std == true
        y_vars = 1.0 / regressor._beta .+ [transpose(PhiT[:, i]) * w_cov * PhiT[:, i] for i in 1:N];
        y_std = sqrt.(y_vars);

        return y, y_std
    end

    return y
end