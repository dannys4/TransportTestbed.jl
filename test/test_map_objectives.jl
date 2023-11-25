function TestFakeObjectives()
    q = FakeQuadRule()
    @test_throws NotImplementedError TransportTestbed.GetQuad(q)
    ell = FakeLossFunction()
    @test_throws NotImplementedError Loss(ell, FakeTransport(), q)
    @test_throws NotImplementedError LossParamGrad(ell, FakeTransport(), q)
end

lognormpdf = x -> -(x^2 + log(2pi)) / 2
gradlognormpdf = x -> -x
vec_lognormpdf = vec -> map(lognormpdf, vec)
vec_gradlognormpdf = vec -> map(gradlognormpdf, vec)
gaussprobhermite = n -> gausshermite(n, normalize = true)

function TestKLDiv(rng::AbstractRNG)
    id = IdMapParam()
    max_order = 0
    linmap = LinearMap(id, max_order + 1)
    alpha = 2.0
    SetParams(linmap, [alpha])
    kl = KLDiv(vec_lognormpdf)
    num_quad_GH = 100
    num_quad_MC = 10_000
    qrule_GH = BlackboxQuad(gaussprobhermite, num_quad_GH)
    qrule_MC = MCQuad(randn(rng, num_quad_MC))
    kl_eval_GH = Loss(kl, linmap, qrule_GH)
    kl_eval_MC = Loss(kl, linmap, qrule_MC) / num_quad_MC

    # For identity map and gaussian reference, kl should evaluate the entropy of the gaussian
    gauss_entropy = (log(2 * pi) - log(alpha * alpha) + alpha * alpha) / 2
    @test abs(kl_eval_GH - gauss_entropy) / gauss_entropy < 1e-8
    @test abs(kl_eval_MC - gauss_entropy) / gauss_entropy < 10 / sqrt(num_quad_MC)
end

function TestRegularizers()
    id = IdMapParam()
    max_order = 2
    linmap = LinearMap(id, max_order + 1)
    default_params = [2.0, 3.0, 4.0]
    SetParams(linmap, default_params)

    param_loss = ParamL2Reg()
    num_quad_GH = 100
    qrule_GH = BlackboxQuad(gaussprobhermite, num_quad_GH)
    param_loss_eval = Loss(param_loss, linmap, qrule_GH)
    @test abs(param_loss_eval - norm(default_params)^2) < 1e-14

    sob_loss = Sobolev12Reg()
    sob_loss_eval = Loss(sob_loss, linmap, qrule_GH)
    # If $T(x) = ax$, then
    # $\int (T(x)^2 + T^\prime(x)^2) p(x) dx =
    # = \int (a^2 x^2 + a^2)p(x) dx = a^2 (Var[x] + 1) = 2a^2$
    # if a^2 is the squared sum of the params and x~N(0,1)
    sob_loss_exact = 2 * (sum(default_params)^2)
    @test abs(sob_loss_eval - sob_loss_exact) < 1e-10

    kl = KLDiv(vec_lognormpdf)
    kl_eval = Loss(kl, linmap, qrule_GH)
    k_weight = 0.5
    p_weight = 0.25
    s_weight = 0.75
    loss = CombinedLoss(kl,
        param_loss,
        sob_loss;
        weight_primary = k_weight,
        weight_param_reg = p_weight,
        weight_sobolev_reg = s_weight,)
    loss_eval = Loss(loss, linmap, qrule_GH)
    exact_loss = kl_eval * k_weight + param_loss_eval * p_weight + sob_loss_eval * s_weight
    @test abs(loss_eval - exact_loss) < 1e-10
end

function TestLossParamGrad(fd_delta = 1e-5)
    linmap = DefaultSigmoidMap()
    num_params = NumParams(linmap)
    default_params = collect(1.0:num_params)
    SetParams(linmap, default_params)
    kl = KLDiv(vec_lognormpdf, vec_gradlognormpdf)
    pr = ParamL2Reg()
    sr = Sobolev12Reg()
    k_weight = 0.5
    p_weight = 0.25
    s_weight = 0.75
    cl = CombinedLoss(kl,
        pr,
        sr;
        weight_primary = k_weight,
        weight_param_reg = p_weight,
        weight_sobolev_reg = s_weight,)
    num_quad_GH = 100
    qrule_GH = BlackboxQuad(gaussprobhermite, num_quad_GH)
    losses = [kl, pr, sr, cl]
    for (j, loss) in enumerate(losses)
        SetParams(linmap, default_params)
        loss_eval = Loss(loss, linmap, qrule_GH)
        loss_pgrad = LossParamGrad(loss, linmap, qrule_GH)
        loss_fd_fix = zeros(length(default_params))
        for j in eachindex(default_params)
            new_params = copy(default_params)
            new_params[j] += fd_delta
            SetParams(linmap, new_params)
            loss_fd_fix[j] = Loss(loss, linmap, qrule_GH)
        end
        fd_diff = (loss_fd_fix .- loss_eval) / fd_delta
        max_err = maximum(abs.(loss_pgrad - fd_diff))
        @test all(max_err .< 10 * fd_delta)
    end
end
