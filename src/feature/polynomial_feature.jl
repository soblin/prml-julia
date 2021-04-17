struct PolynomialFeature
    _degree::Int64
end

function transform(feature::PolynomialFeature, x::AbstractArray{Float64,1})
    # return the design matrix [phi(x_1), phi(x_2), ,,, phi(x_N)]^T of size (n_samples, n_features)
    n_samples = size(x)[1]
    degree = feature._degree
    Phi = zeros(Float64, n_samples, degree + 1)
    for i = 1:n_samples
        Phi[i, :] = [x[i]^j for j = 0:degree]
    end

    return Phi
end
