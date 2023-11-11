function TestSigmoidParamEvaluation()
    f = CreateSigmoidParam(;mapLB = 0., mapUB = 0.)
    xgrid = -5:0.05:5
    line = EvaluateAll(f, 2, xgrid)
    @test all(line[1,:] .== 1.)
    @test sum(abs.(line[2,:] + line[3,:] - xgrid)) < 1e-10

    centers = [[0], [-0.5, 0.5], [-1, 0, 1]]
    support_bound = 100.
    xgrid = vcat([-support_bound, 0., support_bound], xgrid)
    f = CreateSigmoidParam(;centers)
    evaluations = EvaluateAll(f, 5, xgrid)
    @test all(evaluations[1,:] .== 1.)
    @test all(evaluations[4:end,:] .>= 0) && all(evaluations[4:end,:] .<= 1)
    @test abs(evaluations[4,1]) < 1e-8          # Zero on left tail
    @test abs(evaluations[4,2] - 0.5) < 1e-8    # 0.5 at origin
    @test abs(evaluations[4,3] - 1) < 1e-8      # One at right tail
    grid_evals = evaluations[:,4:end]
    diff_first_eval = grid_evals[4:end,2:end] - grid_evals[4:end,1:end-1]
    @test all(diff_first_eval .>= 0.) # Check if monotone
end

function TestSigmoidParamDerivative()
    f = CreateSigmoidParam(;mapLB = 0., mapUB = 0.)
    xgrid = -3:0.05:3
    line, line_diff = Derivative(f, 2, xgrid)
    @test all(line[1,:] .== 1.)
    @test sum(abs.(line[2,:] + line[3,:] - xgrid)) < 1e-10
    @test all(line_diff[1,:] .== 0.)
    @test sum(abs.(line_diff[2,:] + line_diff[3,:] .- 1)) .< 1e-10

    centers = [[0], [-0.5, 0.5], [-1, 0, 1]]
    support_bound = 100.
    fd_delta = 1e-7
    xgrid = vcat([-support_bound, 0., support_bound], xgrid)
    f = CreateSigmoidParam(;centers)
    f_eval = EvaluateAll(f, 5, xgrid)
    f_eval_deriv, diff = Derivative(f, 5, xgrid)
    f_eval_fd = EvaluateAll(f, 5, xgrid .+ fd_delta)
    @test all(abs.(f_eval - f_eval_deriv) .< 1e-14)
    @test all(diff[1,:] .== 0.)

    fd_diff = (f_eval_fd - f_eval)/fd_delta    
    err = abs.(fd_diff - diff)
    @test all(err .< 10*fd_delta)
end

function TestSigmoidParamSecondDerivative()
    f = CreateSigmoidParam(;mapLB = 0., mapUB = 0.)
    xgrid = -3:0.05:3
    line1 = EvaluateAll(f,2,xgrid)
    _, line2_diff = Derivative(f,2,xgrid)
    line, line_diff, line_diff2 = SecondDerivative(f, 2, xgrid)
    @test all(abs.(line - line1) .< 1e-14)
    @test all(abs.(line_diff - line2_diff) .< 1e-14)
    @test all(abs.(line_diff2[1,:]) .< 1e-14)
    @test all(abs.(line_diff2[2,:]+line_diff2[3,:]) .< 1e-14)

    centers = [[0], [-0.5, 0.5], [-1, 0, 1]]
    support_bound = 100.
    fd_delta = 1e-7
    xgrid = vcat([-support_bound, 0., support_bound], xgrid)
    f = CreateSigmoidParam(;centers)
    f_eval = EvaluateAll(f, 5, xgrid)
    _, diff    = Derivative(f, 5, xgrid)
    _, diff_fd = Derivative(f, 5, xgrid .+ fd_delta)
    f_eval_deriv2, diff_deriv2, diff2 = SecondDerivative(f, 5, xgrid)
    @test all(abs.(f_eval - f_eval_deriv2) .< 1e-14)
    @test all(abs.(diff - diff_deriv2) .< 1e-14)
    @test all(diff2[1,:] .== 0.)

    fd_diff2 = (diff_fd - diff)/fd_delta
    err = abs.(fd_diff2 - diff2)
    @test all(err .< 10*fd_delta)
end