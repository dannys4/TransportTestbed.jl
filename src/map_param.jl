function __notImplementParam(method, T)
    __notImplement(method, T, MapParam)
end

function EvaluateAll(p::MapParam, ::Int, ::AbstractVector{<:Real})
    __notImplementParam(EvaluateAll, typeof(p))
end
function Derivative(p::MapParam, ::Int, ::AbstractVector{<:Real})
    __notImplementParam(Derivative, typeof(p))
end
function SecondDerivative(p::MapParam, ::Int, ::AbstractVector{<:Real})
    __notImplementParam(SecondDerivative, typeof(p))
end

Evaluate(p::SigmoidType, ::Float64) = __notImplement(Evaluate, typeof(p), SigmoidType)
Evaluate(p::TailType, ::Float64) = __notImplement(Evaluate, typeof(p), TailType)
Derivative(p::SigmoidType, ::Float64) = __notImplement(Derivative, typeof(p), SigmoidType)
Derivative(p::TailType, ::Float64) = __notImplement(Derivative, typeof(p), TailType)
function SecondDerivative(p::SigmoidType, ::Float64)
    __notImplement(SecondDerivative, typeof(p), SigmoidType)
end
function SecondDerivative(p::TailType, ::Float64)
    __notImplement(SecondDerivative, typeof(p), TailType)
end

function Evaluate(::Type{Logistic}, pt::Float64)
    pt < 0 ? exp(pt) / (1 + exp(pt)) : 1 / (1 + exp(-pt))
end
function Derivative(::Type{Logistic}, pt::Float64)
    f = Evaluate(Logistic, pt)
    f * (1 - f)
end
function SecondDerivative(::Type{Logistic}, pt::Float64)
    f = Evaluate(Logistic, pt)
    f * (1 - f) * (1 - 2 * f)
end

tail_width = 5.0

function Evaluate(::Type{SoftPlus}, pt::Float64)
    pt < 0 ? log(1 + exp(pt)) : -log(exp(-pt) / (1 + exp(-pt)))
end
Derivative(::Type{SoftPlus}, pt::Float64) = Evaluate(Logistic, pt)
SecondDerivative(::Type{SoftPlus}, pt::Float64) = Derivative(Logistic, pt)

function CreateSigmoidParam(;
    centers::__VVN=nothing,
    widths::__VVN=nothing,
    weights::__VVN=nothing,
    mapLB::Real=-5.0,
    mapUB::Real=5.0,
    left_tailwidth::Real=1.0,
    right_tailwidth::Real=1.0,
)
    SigmoidParam{Logistic,SoftPlus}(
        centers, widths, weights, mapLB, mapUB, left_tailwidth, right_tailwidth
    )
end
NumParams(f::SigmoidParam) = f.max_order + 1

function EvaluateAll(
    f::SigmoidParam{T,U}, max_order::Int, pts::AbstractVector{<:Real}
) where {T,U}
    output = Matrix{Float64}(undef, max_order + 1, length(pts))
    @assert max_order >= 2 "Required to evaluate at least constant and tails"
    @assert max_order <= f.max_order "Can only evaluate on through order $(f.max_order)"
    for (p, pt) in enumerate(pts) # TODO: Parallel
        pt = pts[p]
        output[1, p] = 1
        output[2, p] = -Evaluate(U, -f.left_tailwidth * (pt - f.mapLB))
        output[3, p] = Evaluate(U, f.right_tailwidth * (pt - f.mapUB))
        for order in 4:(max_order + 1)
            output[order, p] = 0.0
            order_idx = order - 3
            for j in 1:(order - 3)
                weight_j = f.weights[order_idx][j]
                width_j = f.widths[order_idx][j]
                center_j = f.centers[order_idx][j]
                eval_j = Evaluate(T, width_j * (pt - center_j))
                output[order, p] += weight_j * eval_j
            end
        end
    end
    output
end

function Derivative(
    f::SigmoidParam{T,U}, max_order::Int, pts::AbstractVector{<:Real}
) where {T,U}
    output = Matrix{Float64}(undef, max_order + 1, length(pts))
    diff = Matrix{Float64}(undef, max_order + 1, length(pts))
    @assert max_order >= 2 "Required to evaluate at least constant and tails"
    @assert max_order <= f.max_order "Can only evaluate on through order $(f.max_order)"
    for (p, pt) in enumerate(pts) # TODO: Parallel
        pt = pts[p]
        output[1, p] = 1
        output[2, p] = -Evaluate(U, -f.left_tailwidth * (pt - f.mapLB))
        output[3, p] = Evaluate(U, f.right_tailwidth * (pt - f.mapUB))
        diff[1, p] = 0.0
        diff[2, p] = f.left_tailwidth * Derivative(U, -f.left_tailwidth * (pt - f.mapLB))
        diff[3, p] = f.right_tailwidth * Derivative(U, f.right_tailwidth * (pt - f.mapUB))
        for order in 4:(max_order + 1)
            output[order, p] = 0.0
            diff[order, p] = 0.0
            order_idx = order - 3
            for j in 1:(order - 3)
                weight_j = f.weights[order_idx][j]
                width_j = f.widths[order_idx][j]
                center_j = f.centers[order_idx][j]
                eval_j = Evaluate(T, width_j * (pt - center_j))
                diff_j = width_j * Derivative(T, width_j * (pt - center_j))
                output[order, p] += weight_j * eval_j
                diff[order, p] += weight_j * diff_j
            end
        end
    end
    output, diff
end

function SecondDerivative(
    f::SigmoidParam{T,U}, max_order::Int, pts::AbstractVector{<:Real}
) where {T,U}
    output = Matrix{Float64}(undef, max_order + 1, length(pts))
    diff = Matrix{Float64}(undef, max_order + 1, length(pts))
    diff2 = Matrix{Float64}(undef, max_order + 1, length(pts))
    @assert max_order >= 2 "Required to evaluate at least constant and tails"
    @assert max_order <= f.max_order "Can only evaluate on through order $(f.max_order)"
    for (p, pt) in enumerate(pts) # TODO: Parallel
        pt = pts[p]
        output[1, p] = 1
        output[2, p] = -Evaluate(U, -f.left_tailwidth * (pt - f.mapLB))
        output[3, p] = Evaluate(U, f.right_tailwidth * (pt - f.mapUB))
        diff[1, p] = 0.0
        diff[2, p] = f.left_tailwidth * Derivative(U, -f.left_tailwidth * (pt - f.mapLB))
        diff[3, p] = f.right_tailwidth * Derivative(U, f.right_tailwidth * (pt - f.mapUB))
        diff2[1, p] = 0.0
        diff2[2, p] =
            -f.left_tailwidth *
            f.left_tailwidth *
            SecondDerivative(U, -f.left_tailwidth * (pt - f.mapLB))
        diff2[3, p] =
            f.right_tailwidth *
            f.right_tailwidth *
            SecondDerivative(U, f.right_tailwidth * (pt - f.mapUB))
        for order in 4:(max_order + 1)
            output[order, p] = 0.0
            diff[order, p] = 0.0
            diff2[order, p] = 0.0
            order_idx = order - 3
            for j in 1:(order - 3)
                weight_j = f.weights[order_idx][j]
                width_j = f.widths[order_idx][j]
                center_j = f.centers[order_idx][j]
                eval_j = Evaluate(T, width_j * (pt - center_j))
                diff_j = width_j * Derivative(T, width_j * (pt - center_j))
                diff2_j = width_j * width_j * SecondDerivative(T, width_j * (pt - center_j))
                output[order, p] += weight_j * eval_j
                diff[order, p] += weight_j * diff_j
                diff2[order, p] += weight_j * diff2_j
            end
        end
    end
    output, diff, diff2
end
