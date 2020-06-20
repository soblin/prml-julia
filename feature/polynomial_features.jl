struct PolynomialFeature
    _degree::Int64
end

function transform(feature::PolynomialFeature, x::Array{Float64, 1})
    # return [phi(x_1), phi(x_2),,, phi(x_N)]
    # phi(x_i) is the transfomed feature vector(vertical)
    sample_size = size(x)[1];
    degree = feature._degree;
    ret = zeros(Float64, degree+1, sample_size);
    for i in 1:sample_size
        ret[:, i] = [ x[i]^(j+1) for j in 0:degree];
    end

    return ret;
end
