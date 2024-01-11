include("gmm_base.jl")

function GetQMC_Normal(N::Int, sobol_gen)
    unif_samp = reduce(vcat, Sobol.next!(sobol_gen) for _ in 1:N)
    norminvcdf.(unif_samp)
end

function TrainMap_PosConstraint_Adapt!(
    alg, umap, N_quad_start, loss; verbose::Bool=true, use_adapt=true
)
    num_params = NumParams(umap)
    sobol_gen = SobolSeq(1)
    if use_adapt
        skip(sobol_gen, 2 * N_quad_start)
    else
        skip(sobol_gen, N_quad_start)
    end
    
    qrule = MCQuad(GetQMC_Normal(N_quad_start, sobol_gen))
    next_qrule = MCQuad(GetQMC_Normal(N_quad_start, sobol_gen))
    target_eval_count = 0
    gradtarget_eval_count = 0
    grad_tol = 0.01
    loss_tol = 0.01
    grad_no_adapt_thresh = 5.0
    vars = [
        umap,
        qrule,
        loss,
        -1,
        next_qrule,
        sobol_gen,
        grad_no_adapt_thresh,
        grad_tol,
        loss_tol,
    ]
    function EvalLoss(params::Vector{Float64}, p)
        umap_e, qrule_e, loss_e = p
        target_eval_count += NumQuad(qrule_e)
        SetParams(umap_e, params)
        Loss(loss_e, umap_e, qrule_e)
    end

    function EvalLossGrad!(g::Vector{Float64}, params::Vector{Float64}, p)
        umap_e, qrule_e, loss_e = p
        gradtarget_eval_count += NumQuad(qrule_e)
        SetParams(umap_e, params)
        g .= LossParamGrad(loss_e, umap_e, qrule_e)
        p[4] = norm(g)
        g
    end

    # Increase number of points as needed
    function AdaptCallback(_, lossval)
        umap_e,
        qrule_e,
        loss_e,
        grad_norm,
        next_qrule_e,
        sobol_gen_e,
        grad_no_adapt_thresh,
        grad_tol,
        loss_tol = vars
        # @info "" grad_norm grad_no_adapt_thresh grad_tol
        grad_norm > grad_no_adapt_thresh && return false
        error_check = Loss(loss_e, umap_e, next_qrule_e)
        target_eval_count += NumQuad(next_qrule_e)
        # Approx loss is (error_check + lossval)/2
        # Check |error_check - lossval| relative to approx loss
        # Range is between 0 and 2, with 2 being the worst. Shift and scale accordingly
        loss_diff = 1 - abs(error_check - lossval) / abs(error_check + lossval)
        loss_diff > (1 - loss_tol / 2) && grad_norm < grad_tol / 2 && return true
        # @info "" loss_diff loss_tol
        if grad_norm < grad_tol || loss_diff < loss_tol
            prev_samp = qrule_e.samples
            new_samp = next_qrule_e.samples
            cat_samp = vcat(prev_samp, new_samp)
            vars[2] = MCQuad(cat_samp)
            vars[5] = MCQuad(GetQMC_Normal(length(cat_samp), sobol_gen_e))
            vars[end - 2] *= 0.5
            # vars[end-1] *= 0.7
        end
        return false
    end
    func = OptimizationFunction(EvalLoss; grad=EvalLossGrad!)
    lb = zeros(num_params)
    lb[1] = -Inf
    ub = fill(Inf, num_params)
    prob = OptimizationProblem(func, ones(num_params), vars; lb, ub)
    verbose && @info "Start QMC count: $(NumQuad(vars[2]))"
    callback = use_adapt ? AdaptCallback : (p, l, args...) -> false
    sol = solve(prob, alg; callback)
    verbose &&
        @info "target evals: $(target_eval_count), grad target evals: $(gradtarget_eval_count), Final QMC count: $(NumQuad(vars[2]))"
    params = sol.u
    verbose && (@info "" params)
    SetParams(umap, params)
    target_eval_count, gradtarget_eval_count, NumQuad(vars[2])
end

function DefaultAdapt(
    rng::AbstractRNG,
    target_dist::Distribution;
    alg=LBFGS(),
    umap=DefaultMap(),
    N_quad_start=4,
    L2_reg=7e-3,
    sob_reg=1e-2,
    N_error_pts=10_000,
    use_adapt=true,
    verbose=false,
)
    kl_loss = KLDiv(x -> logpdf.(target_dist, x), x -> gradlogpdf.(target_dist, x))
    loss_obj = CombinedLoss(kl_loss; weight_param_reg=L2_reg, weight_sobolev_reg=sob_reg)
    target_eval_count, gradtarget_eval_count, qmc_count = TrainMap_PosConstraint_Adapt!(
        alg, umap, N_quad_start, loss_obj; use_adapt, verbose
    )
    eval_pts = randn(rng, N_error_pts)
    err = KolmogorovSmirnov(umap, target_dist, eval_pts)
    @info "" target_eval_count gradtarget_eval_count qmc_count err
    umap
end