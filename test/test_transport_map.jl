function TestFakeTransportMap()
    umap = FakeTransport()
    @test_throws NotImplementedError EvaluateMap(umap, [1.,2.,3.])
end

function TestLogDeterminant(umap::TransportMap, rng::AbstractRNG, N_pts = 1000, fd_delta = 1e-5)
    default_params = ones(NumParams(umap))
    SetParams(umap, default_params)
    points = randn(rng, N_pts)
    eval = EvaluateMap(umap, points)
    eval_fd = EvaluateMap(umap, points .+ fd_delta)
    fd_logdet = log.(abs.(eval_fd - eval)/fd_delta)
    eval_logdet = LogDeterminant(umap, points)
    @test all(abs.(eval_logdet - fd_logdet) .< 10*fd_delta)

    eval_fd2 = EvaluateMap(umap, points .+ 2fd_delta)
    fd_logdet2 = log.(abs.(eval_fd2 - eval_fd)/fd_delta)
    fd_logdet_diff = (fd_logdet2 - fd_logdet)/fd_delta
    igrad_logdet = LogDeterminantInputGrad(umap, points)
    @test all(abs.(fd_logdet_diff - igrad_logdet) .< 10*fd_delta)
    
    fd_logdet_pdiff = Matrix{Float64}(undef, NumParams(umap), N_pts)
    for j in 1:NumParams(umap)
        new_params = copy(default_params)
        new_params[j] += fd_delta
        SetParams(umap, new_params)
        fd_logdet_pdiff[j,:] .= LogDeterminant(umap, points)
    end
    SetParams(umap, default_params)
    fd_logdet_pdiff = (fd_logdet_pdiff .- eval_logdet') / fd_delta
    pgrad_logdet = LogDeterminantParamGrad(umap, points)
    @test all(abs.(fd_logdet_pdiff - pgrad_logdet) .< 10*fd_delta)
end