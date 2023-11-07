import TransportTestbed: EvaluateAll, Derivative, SecondDerivative
struct IdMapParam <: TransportTestbed.MapParam end
function TransportTestbed.EvaluateAll(::IdMapParam, max_order::Int, pts::AbstractVector{<:Real})
    output = Matrix{Float64}(undef, max_order+1, length(pts))
    for j in 1:max_order+1
        output[j,:] = pts
    end
    output
end

function TransportTestbed.Derivative(::IdMapParam, max_order::Int, pts::AbstractVector{<:Real})
    output = Matrix{Float64}(undef, max_order+1, length(pts))
    diff = similar(output)
    for j in 1:max_order+1
        output[j,:] = pts
        diff[j,:] .= 1.
    end
    output, diff
end

function TransportTestbed.SecondDerivative(::IdMapParam, max_order::Int, pts::AbstractVector{<:Real})
    output = Matrix{Float64}(undef, max_order+1, length(pts))
    diff = similar(output)
    diff2 = similar(output)
    for j in 1:max_order+1
        output[j,:] = pts
        diff[j,:] .= 1.
        diff2[j,:] .= 0.
    end
    output, diff, diff2
end

function TestIdentityMap()
    id = IdMapParam()
    pts = -5:0.05:5
    max_order = 5
    out1 = EvaluateAll(id, max_order, pts)
    out2, out2_diff = Derivative(id, max_order, pts)
    out3, out3_diff, out3_diff2 = SecondDerivative(id, max_order, pts)
    @test all(out1' .== pts)
    @test all(out2' .== pts)
    @test all(out3' .== pts)
    @test all(out2_diff .== 1)
    @test all(out3_diff .== 1)
    @test all(out3_diff2 .== 0.)
end

function TestLinearIdentityMap()
    id = IdMapParam()
    rng = Xoshiro(29302)
    N_pts = 1000
    points = randn(rng, N_pts)
    num_coeffs = 4
    linmap = LinearMap(id, num_coeffs)
    sample_coeffs = 1:num_coeffs
    SetCoeffs(linmap, sample_coeffs)
    @test all(GetCoeffs(linmap) .== sample_coeffs)
    eval_pts = Evaluate(linmap, points)
    # Evaluations
    @test size(eval_pts) == (N_pts,)
    scale, shift = 0.75, 0.25
    SetCoeffs(linmap, scale*sample_coeffs .+ shift)
    eval_pts_lin = Evaluate(linmap, points)
    @test all(abs.(eval_pts_lin - (scale*eval_pts + EvaluateAll(id, num_coeffs-1, points)'*fill(shift, num_coeffs))) .< 1e-10)

    # Derivatives
    SetCoeffs(linmap, ones(num_coeffs))
    eval_pts_lin, grad_pts_lin = EvaluateInputGrad(linmap, points)
    eval_pts_lin1 = Evaluate(linmap, points)
    @test all(grad_pts_lin .== num_coeffs)
    @test all(eval_pts_lin .== eval_pts_lin1)
    # eval_pts_lin, grad_pts_lin, hess_pts_lin = EvaluateInputGradHess
end