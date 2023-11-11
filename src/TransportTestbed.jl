module TransportTestbed
using Random, LinearAlgebra

abstract type SigmoidType end
abstract type Logistic <: SigmoidType end

abstract type TailType end
abstract type SoftPlus <: TailType end
abstract type MapParam end

module DerivativeFlags
    @enum __DerivativeFlags None=0 ParamGrad=1 InputGrad=2 MixedGrad=3 MixedHess=4 InputHess=5
end

__VV = Vector{<:Vector{<:Any}}
__VT = Vector{Vector{Float64}}
__VVN = Union{__VV, Nothing}
__VR = AbstractVector{<:Real}
__VF = AbstractVector{DerivativeFlags.__DerivativeFlags}

struct SigmoidParam{T<:SigmoidType,U<:TailType} <: MapParam
    centers::__VT
    widths::__VT
    weights::__VT
    max_order::Int
    mapLB::Float64
    mapUB::Float64
    function SigmoidParam{T1,U1}(centers::__VVN, widths::__VVN, weights::__VVN, mapLB, mapUB) where {T1,U1}
        if isnothing(centers)
            centers = Vector{Vector{Float64}}[]
        end
        if isnothing(widths)
            widths = [ones(j) for j in 1:length(centers)]
        end
        if isnothing(weights)
            weights = [ones(j)/j for j in 1:length(centers)]
        end
        if !(length(centers) == length(widths) && length(centers) == length(weights))
            throw(ArgumentError("centers, widths, and weights must all have the same length"))
        end
        for j in eachindex(centers)
            for vec in (centers[j], widths[j], weights[j])
                if length(vec) != j
                    throw(ArgumentError("The jth argument of any nonempty input should be of length j"))
                end
            end
        end
        if mapLB > mapUB
            throw(ArgumentError("mapLB must be smaller than mapUB"))
        end
        new{T1,U1}(centers, widths, weights, length(centers)+2, mapLB, mapUB)
    end
end

struct LinearMap{T<:MapParam}
    __map_eval::T
    __coeff::Vector{Float64}
    function LinearMap(map_eval::U, basisLen::Int) where {U<:MapParam}
        new{U}(map_eval, zeros(basisLen))
    end
end

include("map_param.jl")
include("linear_map.jl")

export SigmoidParam, CreateSigmoidParam, Logistic, SoftPlus
export LinearMap, GetCoeffs, SetCoeffs, NumCoeffs
export EvaluateMap, DerivativeFlags
end