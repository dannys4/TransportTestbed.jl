include("gmm_base.jl")

function TrainMap_PosConstraint_Adapt!(alg, umap, N_quad_start, loss, verbose::Bool = true)
    num_params = NumParams(umap)
    sobol_gen = SobolSeq(1)
    skip(sobol_gen, 2*N_quad_start)
    qrule = MCQuad(reduce(vcat, Sobol.next!(sobol_gen) for _ in N_quad_start))
    next_qrule = MCQuad(reduce(vcat, Sobol.next!(sobol_gen) for _ in N_quad_start))
    target_eval_count = 0
    gradtarget_eval_count = 0
    grad_tol = loss_tol = 0.1
    grad_no_adapt_thresh = 0.5
    vars = [umap, qrule, loss, -1, next_qrule, sobol_gen, grad_no_adapt_thresh, grad_tol, loss_tol]
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
        umap_e, qrule_e, loss_e, grad_norm, next_qrule_e, sobol_gen_e, grad_no_adapt_thresh, grad_tol, loss_tol = vars
        grad_norm > grad_no_adapt_thresh && return false
        error_check = Loss(loss_e, umap_e, next_qrule_e)
        target_eval_count += NumQuad(next_qrule_e)
        # Approx loss is (error_check + lossval)/2
        # Check |error_check - lossval| relative to approx loss
        # Range is between 0 and 2, with 2 being the worst. Shift and scale accordingly
        loss_diff = 1-abs(error_check - lossval)/abs(error_check + lossval)
        if grad_norm < grad_tol || loss_diff < loss_tol
            prev_samp = qrule_e.samples
            new_samp = next_qrule_e.samples
            cat_samp = vcat(prev_samp, new_samp)
            unif_samp = reduce(vcat, Sobol.next!(sobol_gen_e) for _ in 1:length(cat_samp))
            vars[2] = MCQuad(cat_samp)
            vars[5] = MCQuad(norminvcdf.(unif_samp))
        end
        return false
    end
    func = OptimizationFunction(EvalLoss; grad=EvalLossGrad!)
    lb = zeros(num_params)
    lb[1] = -Inf
    ub = fill(Inf, num_params)
    prob = OptimizationProblem(func, ones(num_params), vars; lb, ub)
    @info "Start QMC count: $(NumQuad(vars[2]))"
    sol = solve(prob, alg, callback=AdaptCallback)
    @info "target evals: $(target_eval_count), grad target evals: $(gradtarget_eval_count), Final QMC count: $(NumQuad(vars[2]))"
    params = sol.u
    verbose && (@info "" params)
    SetParams(umap, params)
end

function DefaultAdapt(target_dist::Distribution; alg = LBFGS(), umap = DefaultMap(), N_quad_start = 4, L2_reg = 7e-3, sob_reg = 1e-2)
    kl_loss = KLDiv(x->logpdf.(target_dist, x), x->gradlogpdf.(target_dist, x))
    loss_obj = CombinedLoss(kl_loss, weight_param_reg = L2_reg, weight_sobolev_reg = sob_reg)
    TrainMap_PosConstraint_Adapt!(alg, umap, N_quad_start, loss_obj)
    umap
end