module TransportTestbed
using Random, LinearAlgebra

abstract type SigmoidType end
abstract type Logistic <: SigmoidType end

abstract type TailType end
abstract type SoftPlus <: TailType end

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

__VV = Vector{<:Vector{<:Any}}
__VT = Vector{Vector{Float64}}
__VVN = Union{__VV, Nothing}
struct SigmoidMap{T<:SigmoidType,U<:TailType}
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
export SigmoidMap, CreateSigmoidMap, Logistic, SoftPlus, EvaluateAll, Derivative, SecondDerivative

end