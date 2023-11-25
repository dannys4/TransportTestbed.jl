# Map param functions to pirate
using TransportTestbed: EvaluateAll, Derivative, SecondDerivative

struct FakeParam <: TransportTestbed.MapParam end
struct FakeSigmoid <: TransportTestbed.SigmoidType end
struct FakeTail <: TransportTestbed.TailType end
struct FakeTransport <: TransportTestbed.TransportMap end
struct FakeQuadRule <: TransportTestbed.QuadRule end
struct FakeLossFunction <: TransportTestbed.LossFunction end
struct FakeOptimizer <: TransportTestbed.Optimizer end

struct IdMapParam <: TransportTestbed.MapParam end

function TransportTestbed.EvaluateAll(::IdMapParam,
        max_order::Int,
        pts::AbstractVector{<:Real})
    output = Matrix{Float64}(undef, max_order + 1, length(pts))
    for j in 1:(max_order + 1)
        output[j, :] = pts
    end
    output
end

function TransportTestbed.Derivative(::IdMapParam,
        max_order::Int,
        pts::AbstractVector{<:Real})
    output = Matrix{Float64}(undef, max_order + 1, length(pts))
    diff = similar(output)
    for j in 1:(max_order + 1)
        output[j, :] = pts
        diff[j, :] .= 1.0
    end
    output, diff
end

function TransportTestbed.SecondDerivative(::IdMapParam,
        max_order::Int,
        pts::AbstractVector{<:Real})
    output = Matrix{Float64}(undef, max_order + 1, length(pts))
    diff = similar(output)
    diff2 = similar(output)
    for j in 1:(max_order + 1)
        output[j, :] = pts
        diff[j, :] .= 1.0
        diff2[j, :] .= 0.0
    end
    output, diff, diff2
end

function DefaultIdentityMap(num_params = 4)
    id = IdMapParam()
    linmap = LinearMap(id, num_params)
    SetParams(linmap, ones(num_params))
    linmap
end

function DefaultSigmoidMap(num_sigs = 10)
    centers = [[((2i - 1) - j) / (j + 1) for i in 1:j] for j in 1:num_sigs]
    sigmap = CreateSigmoidParam(; centers)
    LinearMap(sigmap, sigmap.max_order + 1)
end
