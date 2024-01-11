using Pkg
Pkg.activate(dirname(@__DIR__))

using TransportTestbed,
    GLMakie,
    Random,
    Distributions,
    Optimization,
    OptimizationOptimJL,
    FastGaussQuadrature,
    Roots,
    StatsFuns,
    LinearAlgebra

struct BenchPlotArgs
    name::String
    color
    marker::Symbol
end

struct QuadRuleBench
    N::Int
    qrule::TransportTestbed.QuadRule
    error::Ref{Float64}
    name::Symbol
end

struct RegBench
    N::Int
    qrule::TransportTestbed.QuadRule
    error::Ref{Float64}
    l2_reg::Float64
    sob_reg::Float64
end

rng = Xoshiro(392204)
stds = [1.5, 0.5]
means = [-0.5, 0.5]
weights = [0.5, 0.5]

gmm = MixtureModel([Normal(m, s) for (m, s) in zip(means, stds)], weights / sum(weights))
norm_dist = Normal()
map_from_dens = x -> invlogcdf(gmm, logcdf(norm_dist, x))
map_from_samp = x -> invlogcdf(norm_dist, logcdf(gmm, x))

struct gaussprobhermite end

function (gh::gaussprobhermite)(n::Int)
    gausshermite(n, normalize=true)
end

function DefaultMap(;max_degree = 10, mapLB = -0.4, mapUB = 1.6, left_tailwidth = 3, right_tailwidth = 4)
    centers = [norminvcdf.((1:j) / (j + 1)) for j in 1:max_degree]
    map_param = CreateSigmoidParam(; centers, mapLB, mapUB, left_tailwidth, right_tailwidth)
    LinearMap(map_param, NumParams(map_param))
end

function TrainMap_PosConstraint!(alg, umap, qrule, loss, verbose::Bool = true)
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
    verbose && (@info "" params)
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

function KolmogorovSmirnov(
    umap::TransportMap, target_dist::Distribution, eval_pts::AbstractVector
)
    N_pts = length(eval_pts)
    map_eval = EvaluateMap(umap, eval_pts)
    sort!(map_eval)
    e_cdf = collect((1:N_pts) / N_pts)
    true_cdf = cdf(target_dist, map_eval)
    maximum(abs.(e_cdf - true_cdf))
end

function L2(
    umap::TransportMap
)