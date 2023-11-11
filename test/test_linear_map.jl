# Map param functions to pirate
using TransportTestbed: EvaluateAll, Derivative, SecondDerivative
# Internal functions to test evaluation technique
using TransportTestbed: Evaluate, EvaluateInputGrad, EvaluateParamGrad, EvaluateInputParamGrad, EvaluateInputGradHess, EvaluateInputGradMixedGrad, EvaluateParamGradInputGradMixedGradMixedInputHess
struct IdMapParam <: TransportTestbed.MapParam end

# From https://rosettacode.org/wiki/Power_set#Julia
function powerset(x::Vector{T}) where T
    result = Vector{T}[[]]
    for elem in x, j in eachindex(result)
        push!(result, [result[j] ; elem])
    end
    result
end

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

function TestLinearIdentityMapEvaluate()
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

end

function TestLinearIdentityMapGrad()
    id = IdMapParam()
    rng = Xoshiro(29302)
    N_pts = 1000
    fd_delta = 1e-8

    points = randn(rng, N_pts)
    num_coeffs = 4
    linmap = LinearMap(id, num_coeffs)

    # Gradients
    SetCoeffs(linmap, ones(num_coeffs))
    eval_pts_lin1 = Evaluate(linmap, points)
    eval_pts_lin2, grad_pts_lin2 = EvaluateInputGrad(linmap, points)
    @test all(eval_pts_lin2 .== eval_pts_lin1)
    @test all(grad_pts_lin2 .== num_coeffs)
    eval_pts_lin3, grad_pts_lin3 = EvaluateParamGrad(linmap, points)
    @test all(eval_pts_lin3 .== eval_pts_lin1)
    @test size(grad_pts_lin3) == (4,N_pts)
    SetCoeffs(linmap, ones(num_coeffs) .+ fd_delta)
    _, grad_pts_lin3_fd = EvaluateParamGrad(linmap, points)
    @test all((grad_pts_lin3_fd .- grad_pts_lin3)/fd_delta .< num_coeffs*10*fd_delta)
    SetCoeffs(linmap, ones(num_coeffs))
    eval_pts_lin4, igrad_pts_lin4, pgrad_pts_lin4 = EvaluateInputParamGrad(linmap, points)
    @test all(eval_pts_lin4 .== eval_pts_lin1)
    @test all(igrad_pts_lin4 .== grad_pts_lin2)
    @test all(pgrad_pts_lin4 .== grad_pts_lin3)

end

function TestLinearIdentityMapGrad()
    id = IdMapParam()
    rng = Xoshiro(29302)
    N_pts = 1000
    fd_delta = 1e-5

    points = randn(rng, N_pts)
    num_coeffs = 4
    linmap = LinearMap(id, num_coeffs)

    # Gradients
    SetCoeffs(linmap, ones(num_coeffs))
    eval_pts_lin1 = Evaluate(linmap, points)
    eval_pts_lin1_fd = Evaluate(linmap, points .+ fd_delta)
    eval_pts_lin2, grad_pts_lin2 = EvaluateInputGrad(linmap, points)
    igrad_fd_diff = (eval_pts_lin1_fd - eval_pts_lin1)/fd_delta
    @test all(eval_pts_lin2 .== eval_pts_lin1)
    @test all(abs.(grad_pts_lin2 - igrad_fd_diff) .< 10*fd_delta)
    eval_pts_lin3, grad_pts_lin3 = EvaluateParamGrad(linmap, points)
    @test all(eval_pts_lin3 .== eval_pts_lin1)
    @test size(grad_pts_lin3) == (4,N_pts)
    
    eval_pts = Evaluate(linmap, points)
    pgrad_fd_diff = []
    for n in 1:num_coeffs
        new_coeffs = ones(num_coeffs)
        new_coeffs[n] += fd_delta
        SetCoeffs(linmap, new_coeffs)
        eval_pts_param_fd = Evaluate(linmap, points)
        push!(pgrad_fd_diff, (eval_pts_param_fd - eval_pts)/fd_delta)
    end
    SetCoeffs(linmap, ones(num_coeffs))
    pgrad_fd_diff = reduce(hcat, pgrad_fd_diff)'
    
    _, pgrad_pts_lin3 = EvaluateParamGrad(linmap, points)
    @test all(abs.(pgrad_fd_diff .- pgrad_pts_lin3) .< 10*fd_delta)
    
    eval_pts_lin4, igrad_pts_lin4, pgrad_pts_lin4 = EvaluateInputParamGrad(linmap, points)
    @test all(eval_pts_lin4 .== eval_pts_lin1)
    @test all(igrad_pts_lin4 .== grad_pts_lin2)
    @test all(pgrad_pts_lin4 .== grad_pts_lin3)
end


function TestLinearIdentityMapHess()
    id = IdMapParam()
    rng = Xoshiro(29302)
    N_pts = 1000
    fd_delta = 1e-8

    points = randn(rng, N_pts)
    num_coeffs = 4
    linmap = LinearMap(id, num_coeffs)
    SetCoeffs(linmap, ones(num_coeffs))

    # Hessians
    eval_pts_lin1, igrad_pts_lin1 = EvaluateInputGrad(linmap, points)
    eval_pts_lin2, igrad_pts_lin2, ihess_pts_lin2 = EvaluateInputGradHess(linmap, points)
    _, igrad_pts_lin1_fd = EvaluateInputGrad(linmap, points .+ fd_delta)
    @test all(eval_pts_lin1 .== eval_pts_lin2)
    @test all(igrad_pts_lin1 .== igrad_pts_lin2)
    igrad_fd_diff = (igrad_pts_lin1_fd - igrad_pts_lin1)/fd_delta
    @test all(abs.(igrad_fd_diff - ihess_pts_lin2) .< 10*fd_delta)

    mixed_fd_diff = []
    for n in 1:num_coeffs
        new_coeffs = ones(num_coeffs)
        new_coeffs[n] += fd_delta
        SetCoeffs(linmap, new_coeffs)
        _, grad_pts_param_fd = EvaluateInputGrad(linmap, points)
        push!(mixed_fd_diff, (grad_pts_param_fd - igrad_pts_lin1)/fd_delta)
    end
    SetCoeffs(linmap, ones(num_coeffs))
    mixed_fd_diff = reduce(hcat, mixed_fd_diff)'

    eval_pts_lin3, igrad_pts_lin3, mgrad_pts_lin3 = EvaluateInputGradMixedGrad(linmap, points)
    @test all(eval_pts_lin1 .== eval_pts_lin3)
    @test all(igrad_pts_lin1 .== igrad_pts_lin3)
    @test all(abs.(mixed_fd_diff - mgrad_pts_lin3) .< 10*fd_delta)

    _,_,mgrad_pts_lin3_fd = EvaluateInputGradMixedGrad(linmap, points .+ fd_delta)
    mixed_fd_diff2 = (mgrad_pts_lin3_fd - mgrad_pts_lin3)/fd_delta

    _, pgrad_pts_lin1 = EvaluateParamGrad(linmap, points)
    eval_pts_lin4, pgrad_pts_lin4, igrad_pts_lin4, mgrad_pts_lin4, mhess_pts_lin4, ihess_pts_lin4 = EvaluateParamGradInputGradMixedGradMixedInputHess(linmap, points)
    @test all(eval_pts_lin1 .== eval_pts_lin4)
    @test all(igrad_pts_lin1 .== igrad_pts_lin4)
    @test all(pgrad_pts_lin1 .== pgrad_pts_lin4)
    @test all(ihess_pts_lin2 .== ihess_pts_lin4)
    @test all(mgrad_pts_lin3 .== mgrad_pts_lin4)
    @test all(abs.(mixed_fd_diff2 - mhess_pts_lin4) .< 10*fd_delta)
end

function TestLinearIdentityMapEvaluateMap()
    id = IdMapParam()
    rng = Xoshiro(3984872)
    N_pts = 1000
    points = randn(rng, N_pts)
    num_coeffs = 4
    linmap = LinearMap(id, num_coeffs)
    SetCoeffs(linmap, ones(num_coeffs))

    ref_eval, ref_pgrad, ref_igrad, ref_mgrad, ref_mhess, ref_ihess = EvaluateParamGradInputGradMixedGradMixedInputHess(linmap, points)
    DF = DerivativeFlags
    eval_plus_flags = [(DF.None, ref_eval), (DF.ParamGrad, ref_pgrad), (DF.InputGrad,ref_igrad), (DF.MixedGrad, ref_mgrad), (DF.MixedHess, ref_mhess), (DF.InputHess, ref_ihess)]
    idxs = collect(eachindex(eval_plus_flags))
    pset = powerset(idxs)
    for idx in pset
        idx_eval_flags = eval_plus_flags[idx]
        idx_flags = [eval_flag[1] for eval_flag in idx_eval_flags]
        idx_ref_evals = [eval_flag[2] for eval_flag in idx_eval_flags]
        idx_evals = EvaluateMap(linmap, points, idx_flags)
        for j in eachindex(idx)
            @test all(idx_evals[j] .== idx_ref_evals[j])
        end
    end
end