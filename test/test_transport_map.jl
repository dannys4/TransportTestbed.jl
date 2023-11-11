function TestFakeTransportMap()
    umap = FakeTransport()
    @test_throws NotImplementedError EvaluateMap(umap, [1.,2.,3.])
end

function TestLogDeterminant(umap::TransportMap, rng::AbstractRNG, N_pts = 1000, fd_delta = 1e-5)
    SetParams(umap, ones(NumParams(umap)))
    points = randn(rng, N_pts)
    eval = EvaluateMap(umap, points)
    eval_fd = EvaluateMap(umap, points .+ fd_delta)
    fd_logdet = log.(abs.(eval - eval_fd)/fd_delta)
    eval_logdet = LogDeterminant(umap, points)
    @test all(abs.(eval_logdet - fd_logdet) .< 10*fd_delta)

    for j in 1:NumParams(umap)
        
    end
end