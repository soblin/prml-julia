module polynomial_features

export PolynomialFeature
export transform2Feature

struct PolynomialFeature
    _degree::Int64
end

function transform2Feature(feature::PolynomialFeature, x)
    sample_size = size(x)[1];
    return sample_size
end

end # module
