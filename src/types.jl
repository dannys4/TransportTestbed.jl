## Types for Map evaluation
### Abstract types for Map Evaluation
abstract type SigmoidType end
abstract type Logistic <: SigmoidType end

abstract type TailType end
abstract type SoftPlus <: TailType end
abstract type MapParam end
abstract type TransportMap end

### Enumeration for different derivative flags
module DerivativeFlags
@enum __DerivativeFlags ErrorFlag=0 None=1 ParamGrad=2 InputGrad=3 MixedGrad=4 MixedHess=5 InputHess=6
end

### Concrete types for map evaluation

__VV = AbstractVector{<:AbstractVector{<:Real}}
__VT = Vector{Vector{Float64}}
__VVN = Union{__VV, Nothing}
__VR = AbstractVector{<:Real}
__VF = AbstractVector{DerivativeFlags.__DerivativeFlags}

struct SigmoidParam{T <: SigmoidType, U <: TailType} <: MapParam
    centers::__VT
    widths::__VT
    weights::__VT
    max_order::Int
    mapLB::Float64
    mapUB::Float64
    left_tailwidth::Float64
    right_tailwidth::Float64
    function SigmoidParam{T1, U1}(centers::__VVN,
            widths::__VVN,
            weights::__VVN,
            mapLB::Real,
            mapUB::Real,
            left_tailwidth::Real,
            right_tailwidth::Real) where {T1, U1}
        if isnothing(centers)
            centers = Vector{Vector{Float64}}[]
        end
        if !isa(centers, Vector{Vector{Float64}})
            new_centers = Vector{Vector{Float64}}(undef, length(centers))
            for j in eachindex(centers)
                center = centers[j]
                @assert center isa AbstractVector{<:Real} ""
                new_centers[j] = Float64.(center)
            end
            centers = new_centers
        end
        if isnothing(widths)
            widths = [ones(j) for j in 1:length(centers)]
        end
        if isnothing(weights)
            weights = [ones(j) / j for j in 1:length(centers)]
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
        new{T1, U1}(centers,
            widths,
            weights,
            length(centers) + 2,
            mapLB,
            mapUB,
            left_tailwidth,
            right_tailwidth)
    end
end

struct LinearMap{T <: MapParam} <: TransportMap
    __map_eval::T
    __coeff::Vector{Float64}
    function LinearMap(map_eval::U, basisLen::Int) where {U <: MapParam}
        new{U}(map_eval, zeros(basisLen))
    end
end

## Types for Optimization Objectives
### Abstract types for Optimization objectives
abstract type QuadRule end
abstract type LossFunction end

### Concrete types for Optimization objectives
# Quadrature needs to be more careful in multiple dimensions
struct MCQuad{T} <: QuadRule
    samples::Vector{T}
    function MCQuad(samples::Vector{U}) where {U}
        new{U}(samples)
    end
end

struct BlackboxQuad{T} <: QuadRule
    eval::T
    num_quad::Int
end

struct KLDiv{T, U} <: LossFunction
    logdensity::T
    gradlogdensity::U
    function KLDiv(logdensity::T1, gradlogdensity::U1 = nothing) where {T1, U1}
        new{T1, U1}(logdensity, gradlogdensity)
    end
end
struct ParamL2Reg <: LossFunction end

# Imitates W_{1,2}, i.e. ||T||_{W_{1,2}}^2 = ||T||_{L_2}^2 + ||T'||_{L_2}^2
struct Sobolev12Reg <: LossFunction end

struct CombinedLoss{P, PReg, SReg} <:
       LossFunction where {P <: LossFunction, PReg <: LossFunction, SReg <: LossFunction}
    weight_primary::Float64
    eval_primary::P
    weight_p_reg::Float64
    eval_p_reg::PReg
    weight_sob_reg::Float64
    eval_sob_reg::SReg
    function CombinedLoss(primary::P1,
            param_reg::PReg1 = ParamL2Reg(),
            sob_reg::SReg1 = Sobolev12Reg();
            weight_primary = 1.0,
            weight_param_reg = 0.0,
            weight_sobolev_reg = 0.0,) where {P1, PReg1, SReg1}
        new{P1, PReg1, SReg1}(weight_primary,
            primary,
            weight_param_reg,
            param_reg,
            weight_sobolev_reg,
            sob_reg)
    end
end

## Types for Optimization
### Abstract types for Optimization
abstract type Optimizer end
abstract type LineSearch end

### Concrete types for Optimization
struct Nesterov <: Optimizer
    max_iter::Int
    lr::Float64
    mu::Float64
    # HIGHLY RECOMMENDED to keep lr, mu as fixed
    function Nesterov(max_iter = 1000, lr = 0.0001, mu = 0.95)
        new(max_iter, lr, mu)
    end
end
struct Armijo <: LineSearch end
struct BifidelityType{T, U, V} <: T where {T, U <: T, V <: T}
    lo::U
    hi::V
end

struct TrustRegion{U, V} <: Optimizer where {U, V}
    lo_solver::U
end
