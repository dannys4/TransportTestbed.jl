function TestKLDiv()
    id = IdMapParam()
    max_order = 0
    linmap = LinearMap(id, max_order+1)
    SetParams(linmap, [1.])
    kl = KLDiv()
    normpdf = x -> exp(-x^2/2)/sqrt(2pi)
    vec_lognormpdf = vec -> map(log∘normpdf, vec)
    num_quad = 100
    qrule = BlackboxQuad(TransportTestbed.gaussprobhermite)
    kl_eval = Loss(kl, linmap, vec_lognormpdf, qrule, num_quad)
    # For identity map and gaussian reference, kl should evaluate the entropy of the gaussian
    gauss_entropy = log(sqrt(2*pi*exp(1)))
    
    @test abs(kl_eval - gauss_entropy)/gauss_entropy < 1e-8
end