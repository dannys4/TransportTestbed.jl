module TransportTestbed
using Random, LinearAlgebra

include("utils.jl")
include("types.jl")

include("map_param.jl")
include("linear_map.jl")
include("map_objectives.jl")
include("transport_map.jl")
include("optimizer.jl")
include("quad.jl")

export SigmoidParam, CreateSigmoidParam, Logistic, SoftPlus
export TransportMap,
    LinearMap,
    GetParams,
    SetParams,
    NumParams,
    LogDeterminant,
    LogDeterminantInputGrad,
    LogDeterminantParamGrad
export EvaluateMap, DerivativeFlags
export BlackboxQuad,
    MCQuad, Loss, LossParamGrad, KLDiv, ParamL2Reg, Sobolev12Reg, CombinedLoss
export Optimize, Nesterov, hermite_gk22
end
