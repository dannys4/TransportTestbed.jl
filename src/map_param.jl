
function __notImplement(method,::Type{T} = T, ::Type{U} = U) where {T,U}
    throw(NotImplementedError(U, method, T))
end
function __notImplementParam(method, T1::Type{T} = T) where {T}
    __notImplement{T,MapParam}(method, T1)
end

EvaluateAll(::T, ::Int, ::AbstractVector{<:Real}) where {T<:MapParam} = __notImplementParam{T}(EvaluateAll)
Derivative(::T, ::Int, ::AbstractVector{<:Real}) where {T<:MapParam} = __notImplementParam{T}(Derivative)
SecondDerivative(::T, ::Int, ::AbstractVector{<:Real}) where {T<:MapParam} = __notImplementParam{T}(SecondDerivative)

Evaluate(::Type{T}, ::Float64) where {T<:SigmoidType} = __notImplement{T,SigmoidType}(Evaluate)
Evaluate(::Type{T}, ::Float64) where {T<:TailType} = __notImplement{T,TailType}(Evaluate)
Derivative(::Type{T}, ::Float64) where {T<:SigmoidType} = __notImplement{T,SigmoidType}(Derivative)
Derivative(::Type{T}, ::Float64) where {T<:TailType} = __notImplement{T,TailType}(Derivative)
SecondDerivative(::Type{T}, ::Float64) where {T<:SigmoidType} = __notImplement{T,SigmoidType}(SecondDerivative)
SecondDerivative(::Type{T}, ::Float64) where {T<:TailType} = __notImplement{T,TailType}(SecondDerivative)

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


CreateSigmoidParam(;centers = nothing, widths = nothing, weights=nothing, mapLB=-5., mapUB=5.) = SigmoidParam{Logistic, SoftPlus}(centers, widths, weights, mapLB, mapUB)

function EvaluateAll(f::SigmoidParam{T,U}, max_order::Int, pts::AbstractVector{<:Real}) where {T,U}
    output = Matrix{Float64}(undef, max_order+1, length(pts))
    @assert max_order >= 2 "Required to evaluate at least constant and tails"
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

function Derivative(f::SigmoidParam{T,U}, max_order::Int, pts::AbstractVector{<:Real}) where {T,U}
    output = Matrix{Float64}(undef, max_order+1, length(pts))
    diff   = Matrix{Float64}(undef, max_order+1, length(pts))
    @assert max_order >= 2 "Required to evaluate at least constant and tails"
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

function SecondDerivative(f::SigmoidParam{T,U}, max_order::Int, pts::AbstractVector{<:Real}) where {T,U}
    output  = Matrix{Float64}(undef, max_order+1, length(pts))
    diff    = Matrix{Float64}(undef, max_order+1, length(pts))
    diff2   = Matrix{Float64}(undef, max_order+1, length(pts))
    @assert max_order >= 2 "Required to evaluate at least constant and tails"
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