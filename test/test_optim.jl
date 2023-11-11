using TransportTestbed

function TestKLDiv()
    id = IdMapParam()
    max_order = 0
    linmap = LinearMap(id, max_order)
    SetParams(linmap, [1.])
    kl = KLDiv()
    normpdf = x -> exp(-x^2/2)/sqrt(2pi)
    num_pts = 100
    qrule = BlackboxQuad(TransportTestbed.gaussprobhermite)
    kl_eval = kl(linmap, normpdf, qrule, num_pts)
    # For identity map and gaussian reference, kl should evaluate the entropy of the gaussian
    gauss_entropy = log(sqrt(2*pi*exp(1)))
    # TODO: Enable this
    @test abs(kl_eval - gauss_entropy)/gauss_entropy < 1e-8 skip=true
end