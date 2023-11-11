module TransportTestbed
using Random, LinearAlgebra

abstract type SigmoidType end
abstract type Logistic <: SigmoidType end

abstract type TailType end
abstract type SoftPlus <: TailType end

abstract type MapParam end
EvaluateAll(::MapParam, ::Int, ::AbstractVector{<:Real}) = @error "Evaluate not implemented for this type"
Derivative(::MapParam, ::Int, ::AbstractVector{<:Real}) = @error "Derivative not implemented for this type"
SecondDerivative(::MapParam, ::Int, ::AbstractVector{<:Real}) = @error "SecondDerivative not implemented for this type"

Evaluate(::Type{SigmoidType}, ::Float64) = nothing
Evaluate(::Type{TailType}, ::Float64) = nothing
Derivative(::Type{SigmoidType}, ::Float64) = nothing
Derivative(::Type{TailType}, ::Float64) = nothing
SecondDerivative(::Type{SigmoidType}, ::Float64) = nothing
SecondDerivative(::Type{TailType}, ::Float64) = nothing


Evaluate(::Type{Logistic}, pt::Float64) = pt < 0 ? exp(pt)/(1+exp(pt)) : 1/(1+exp(-pt))
function Derivative(::Type{Logistic}, pt::Float64)
    f = Evaluate(Logistic, pt)
    f*(1-f)
end
function SecondDerivative(::Type{Logistic}, pt::Float64)
    f = Evaluate(Logistic, pt)
    f*(1-f)*(1-2*f)
end

Evaluate(::Type{SoftPlus}, pt::Float64) = log(1+exp(pt))
Derivative(::Type{SoftPlus}, pt::Float64) = Evaluate(Logistic, pt)
SecondDerivative(::Type{SoftPlus}, pt::Float64) = Derivative(Logistic, pt)

module DerivativeFlags
    @enum __DerivativeFlags None=0 ParamGrad=1 InputGrad=2 MixedGrad=3 MixedHess=4 InputHess=5
end

__VV = Vector{<:Vector{<:Any}}
__VT = Vector{Vector{Float64}}
__VVN = Union{__VV, Nothing}
__VR = AbstractVector{<:Real}
__VF = AbstractVector{DerivativeFlags.__DerivativeFlags}

# Calculates the maximum input derivative based on flags
function InputDerivatives(flags::__VF)
    m_flags = UInt8(maximum(flags))
    m_flags < 2 && return 0
    m_flags < 4 && return 1
    return 2
end

struct SigmoidMap{T<:SigmoidType,U<:TailType} <: MapParam
    centers::__VT
    widths::__VT
    weights::__VT
    max_order::Int
    mapLB::Float64
    mapUB::Float64
    function SigmoidMap{T1,U1}(centers::__VVN, widths::__VVN, weights::__VVN, mapLB, mapUB) where {T1,U1}
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
        for j in 1:length(centers)
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

CreateSigmoidMap(;centers = nothing, widths = nothing, weights=nothing, mapLB=-5., mapUB=5.) = SigmoidMap{Logistic, SoftPlus}(centers, widths, weights, mapLB, mapUB)

function EvaluateAll(f::SigmoidMap{T,U}, max_order::Int, pts::AbstractVector{<:Real}) where {T,U}
    output = Matrix{Float64}(undef, max_order+1, length(pts))
    @assert max_order <= f.max_order "Can only evaluate on through order $(f.max_order)"
    for (p,pt) in enumerate(pts) # TODO: Parallel
        pt = pts[p]
        output[1,p] = 1
        output[2,p] = -Evaluate(U, -(pt - f.mapLB))
        output[3,p] =  Evaluate(U,  (pt - f.mapUB))
        for order = 4:max_order+1
            output[order,p] = 0.
            order_idx = order - 3
            for j in 1:order-3
                weight_j = f.weights[order_idx][j]
                width_j = f.widths[order_idx][j]
                center_j = f.centers[order_idx][j]
                eval_j = Evaluate(T, width_j*(pt - center_j))
                output[order,p] += weight_j*eval_j
            end
        end
    end
    output
end

function Derivative(f::SigmoidMap{T,U}, max_order::Int, pts::AbstractVector{<:Real}) where {T,U}
    output = Matrix{Float64}(undef, max_order+1, length(pts))
    diff   = Matrix{Float64}(undef, max_order+1, length(pts))
    @assert max_order <= f.max_order "Can only evaluate on through order $(f.max_order)"
    for (p,pt) in enumerate(pts) # TODO: Parallel
        pt = pts[p]
        output[1,p] = 1
        output[2,p] = -Evaluate(U, -(pt - f.mapLB))
        output[3,p] =  Evaluate(U,  (pt - f.mapUB))
        diff[1,p] = 0.
        diff[2,p] = Derivative(U, -(pt - f.mapLB))
        diff[3,p] = Derivative(U,  (pt - f.mapUB))
        for order = 4:max_order+1
            output[order,p] = 0.
            diff[order,p]   = 0.
            order_idx = order - 3
            for j in 1:order-3
                weight_j = f.weights[order_idx][j]
                width_j = f.widths[order_idx][j]
                center_j = f.centers[order_idx][j]
                eval_j = Evaluate(T, width_j*(pt - center_j))
                diff_j = width_j*Derivative(T, width_j*(pt - center_j))
                output[order,p] += weight_j*eval_j
                diff[order,p] += weight_j*diff_j
            end
        end
    end
    output, diff
end

function SecondDerivative(f::SigmoidMap{T,U}, max_order::Int, pts::AbstractVector{<:Real}) where {T,U}
    output  = Matrix{Float64}(undef, max_order+1, length(pts))
    diff    = Matrix{Float64}(undef, max_order+1, length(pts))
    diff2   = Matrix{Float64}(undef, max_order+1, length(pts))
    @assert max_order <= f.max_order "Can only evaluate on through order $(f.max_order)"
    for (p,pt) in enumerate(pts) # TODO: Parallel
        pt = pts[p]
        output[1,p] = 1
        output[2,p] = -Evaluate(U, -(pt - f.mapLB))
        output[3,p] =  Evaluate(U,  (pt - f.mapUB))
        diff[1,p] = 0.
        diff[2,p] = Derivative(U, -(pt - f.mapLB))
        diff[3,p] = Derivative(U,  (pt - f.mapUB))
        diff2[1,p] = 0.
        diff2[2,p] = -SecondDerivative(U, -(pt - f.mapLB))
        diff2[3,p] =  SecondDerivative(U,  (pt - f.mapUB))
        for order = 4:max_order+1
            output[order,p] = 0.
            diff[order,p]   = 0.
            diff2[order,p]  = 0.
            order_idx = order - 3
            for j in 1:order-3
                weight_j = f.weights[order_idx][j]
                width_j = f.widths[order_idx][j]
                center_j = f.centers[order_idx][j]
                eval_j = Evaluate(T, width_j*(pt - center_j))
                diff_j = width_j*Derivative(T, width_j*(pt - center_j))
                diff2_j = width_j*width_j*SecondDerivative(T, width_j*(pt - center_j))
                output[order,p] += weight_j*eval_j
                diff[order,p] += weight_j*diff_j
                diff2[order,p] += weight_j*diff2_j
            end
        end
    end
    output, diff, diff2
end

struct LinearMap{T<:MapParam}
    __map_eval::T
    __coeff::Vector{Float64}
    function LinearMap(map_eval::U, basisLen::Int) where {U<:MapParam}
        new{U}(map_eval, zeros(basisLen))
    end
end

function SetCoeffs(linmap::LinearMap, coeffs::__VR)
    linmap.__coeff .= coeffs
end

GetCoeffs(linmap::LinearMap) = linmap.__coeff

function Evaluate(f::LinearMap, points::__VR)
    evals = EvaluateAll(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff
end

function EvaluateInputGrad(f::LinearMap, points::__VR)
    evals, deriv = Derivative(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, deriv'f.__coeff
end

function EvaluateParamGrad(f::LinearMap, points::__VR)
    evals = EvaluateAll(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, evals
end

function EvaluateInputParamGrad(f::LinearMap, points::__VR)
    evals, deriv = Derivative(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, deriv'f.__coeff, evals
end

function EvaluateInputGradHess(f::LinearMap, points::__VR)
    evals, diff, diff2 = SecondDerivative(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, diff'f.__coeff, diff2'f.__coeff
end

function EvaluateInputGradMixedGrad(f::LinearMap, points::__VR)
    evals, diff, = Derivative(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, diff'f.__coeff, diff
end

function EvaluateParamGradInputGradMixedGradMixedInputHess(f::LinearMap, points::__VR)
    evals, diff, diff2 = SecondDerivative(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, evals, diff'f.__coeff, diff, diff2, diff2'f.__coeff
end

function EvaluateMap(f::LinearMap, points::__VR, deriv_flags::__VF)
    length(deriv_flags) == 0 && return []
    max_deriv = InputDerivatives(deriv_flags)
    eval = diff = diff2 = nothing
    if max_deriv == 0
        eval = EvaluateAll(f.__map_eval, length(f.__coeff)-1, points)
    elseif max_deriv == 1
        eval, diff = Derivative(f.__map_eval, length(f.__coeff)-1, points)
    elseif max_deriv == 2
        eval, diff, diff2 = SecondDerivative(f.__map_eval, length(f.__coeff)-1, points)
    else
        @error "Invalid number of derivatives, $max_deriv"
    end
    ret = []
    for flag in deriv_flags
        if flag == DerivativeFlags.None
            push!(ret, eval'f.__coeff)
        elseif flag == DerivativeFlags.InputGrad
            push!(ret, diff'f.__coeff)
        elseif flag == DerivativeFlags.InputHess
            push!(ret, diff2'f.__coeff)
        elseif flag == DerivativeFlags.ParamGrad
            push!(ret, eval)
        elseif flag == DerivativeFlags.MixedGrad
            push!(ret, diff)
        elseif flag == DerivativeFlags.MixedHess
            push!(ret, diff2)
        else
            @error "Could not find flag $flag"
        end
    end
    ret
end
EvaluateMap(f::LinearMap, points::__VR, deriv_flag::DerivativeFlags.__DerivativeFlags = DerivativeFlags.None) = EvaluateMap(f, points, [deriv_flag])[]

export SigmoidMap, CreateSigmoidMap, Logistic, SoftPlus
export LinearMap, GetCoeffs, SetCoeffs
export EvaluateMap, DerivativeFlags
end