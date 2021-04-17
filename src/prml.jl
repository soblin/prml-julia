module prml

using Distributions
using SpecialFunctions
using Statistics
using LinearAlgebra
using Plots

export BernoulliDist
export BetaDist
export DirichletDist
export CategoricalDist
export GaussianDist
export GaussianBayesMeanDist
export GaussianBayesVarDist
export GammaDist
export MultivariateGaussianDist

export pdf
export fitting
export draw

export PolynomialFeature
export GaussianFeature
export SigmoidalFeature
export transform

export LinearRegressor
export RidgeRegressor
export BayesianRegressor
export EmpiricalBayesianRegressor
export fitting
export log_evidence
export predict
export predictSampling

export AbstractLayer
export LinearLayer
export SigmoidLayer
export TanhLayer
export ReLULayer
export forward_propagation
export activate_derivative
export backward_propagation

export AbstractCostFunction
export SigmoidCrossEntropy
export SoftmaxCrossEntropy
export SumSquareError
export GaussianMixtureError
export delta
export cost

export NeuralNetwork
export fitting
export predict

export PolynomialKernel
export RBFKernel
export GaussianProcessRegressor
export fitting
export predict


include("distribution/distribution.jl")
include("feature/feature.jl")
include("linear/linear.jl")
include("neural_network/neural_network.jl")
include("kernel/kernel.jl")

end # module
