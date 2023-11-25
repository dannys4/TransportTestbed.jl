using TransportTestbed,
    GLMakie,
    Random,
    Distributions,
    Optimization,
    OptimizationOptimJL,
    FastGaussQuadrature,
    Roots
rng = Xoshiro(392204)
stds = [1.5, 0.5]
means = [-0.5, 0.5]
weights = [0.5, 0.5]

gmm = MixtureModel([Normal(m, s) for (m, s) in zip(means, stds)], weights / sum(weights))
norm_dist = Normal()
map_from_dens = x -> invlogcdf(gmm, logcdf(norm_dist, x))
map_from_samp = x -> invlogcdf(norm_dist, logcdf(gmm, x))

function TrainMap_PosConstraint(alg, umap, qrule, loss)
    vars = (umap, qrule, loss)
    function EvalLoss(params::Vector{Float64}, p)
        umap_e, qrule_e, loss_e = p
        SetParams(umap_e, params)
        Loss(loss_e, umap_e, qrule_e)
    end
    function EvalLossGrad!(g::Vector{Float64}, params::Vector{Float64}, p)
        umap_e, qrule_e, loss_e = p
        SetParams(umap_e, params)
        g .= LossParamGrad(loss_e, umap_e, qrule_e)
    end
    func = OptimizationFunction(EvalLoss; grad=EvalLossGrad!)
    num_params = NumParams(umap)
    lb = zeros(num_params)
    lb[1] = -Inf
    ub = fill(Inf, num_params)
    prob = OptimizationProblem(func, ones(num_params), vars; lb, ub)
    sol = solve(prob, alg)
    params = sol.u
    @info "" params
    SetParams(umap, params)
end

# Courtesy of Seth Axen: https://github.com/JuliaStats/Distributions.jl/issues/1788
function Distributions.gradlogpdf(d::AbstractMixtureModel, x::Float64)
    ps = probs(d)
    ds = components(d)
    inds = filter(i -> !iszero(ps[i]), eachindex(ps, ds))
    ws = Distributions.pweights(
        Distributions.softmax!([log(ps[i]) + logpdf(ds[i], x) for i in inds])
    )
    g = mean(map(i -> gradlogpdf(ds[i], x), inds), ws)
    g
end

# Simple external hybrid inverse implementation
function InverseMap(umap::TransportMap, point::Real; lb::Real=-20, ub::Real=20, maxiters=3)
    f = x -> EvaluateMap(umap, [x])[] - point
    tracks = Roots.Tracks()
    x_rough = find_zero(f, (lb, ub), Bisection(); maxiters, tracks)
    (tracks.convergence_flag != :not_converged) && return x_rough
    fp = x -> EvaluateMap(umap, [x], DerivativeFlags.InputGrad)[]
    x_fine = find_zero((f, fp), x_rough, Roots.Newton())
    x_fine
end

function TransportLogpdf_from_dens(
    umap::TransportMap, reference::Distribution, points::AbstractVector; inverse_kwargs...
)
    imap = point -> InverseMap(umap, point; inverse_kwargs...)
    map_points = imap.(points)
    eval_ref_logpdf = logpdf.(reference, map_points)
    eval_logdet = LogDeterminant(umap, map_points)
    eval_ref_logpdf - eval_logdet
end
