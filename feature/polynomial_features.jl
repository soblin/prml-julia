module polynomial_features

export PolynomialFeature
export transform

struct PolynomialFeature
    _degree::Int64
end

function transform(feature::PolynomialFeature, x)
    sample_size = size(x)[1];
    degree = feature._degree;
    ret = zeros(Float64, degree+1, sample_size);
    for i in 1:(degree+1)
        ret[i, :] = [ j^(i-1) for j in x];
    end

    tmp = transpose(ret);
    ret = collect(reshape(tmp, sample_size, degree+1));
    return ret;
end

end # module
