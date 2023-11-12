function TestFakeOptimization()
    q = FakeQuadRule()
    @test_throws NotImplementedError TransportTestbed.GetQuad(q, 5)
    ell = FakeLossFunction()
    @test_throws NotImplementedError Loss(ell, FakeTransport(), nothing, q, 5)
end

function TestKLDiv(rng::AbstractRNG)
    id = IdMapParam()
    max_order = 0
    linmap = LinearMap(id, max_order+1)
    SetParams(linmap, [1.])
    kl = KLDiv()
    normpdf = x -> exp(-x^2/2)/sqrt(2pi)
    vec_lognormpdf = vec -> map(log∘normpdf, vec)
    num_quad_GH = 100
    num_quad_MC = 10_000
    qrule_GH = BlackboxQuad(TransportTestbed.gaussprobhermite)
    qrule_MC = MCQuad(randn(rng, num_quad_MC))
    kl_eval_GH = Loss(kl, linmap, vec_lognormpdf, qrule_GH, num_quad_GH)
    kl_eval_MC = Loss(kl, linmap, vec_lognormpdf, qrule_MC, num_quad_MC)/num_quad_MC
    # For identity map and gaussian reference, kl should evaluate the entropy of the gaussian
    gauss_entropy = log(sqrt(2*pi*exp(1)))
    
    @test abs(kl_eval_GH - gauss_entropy)/gauss_entropy < 1e-8
    @test abs(kl_eval_MC - gauss_entropy)/gauss_entropy < 10/sqrt(num_quad_MC)
end