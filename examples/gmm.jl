using TransportTestbed,
    GLMakie,
    Random,
    Distributions,
    Optimization,
    OptimizationOptimJL,
    FastGaussQuadrature,
    Roots
rng = Xoshiro(392204)
N_samples = 20_000
stds = [1.5, 0.5]
means = [-0.5, 0.5]
weights = [0.5, 0.5]

##
Ns = [round(Int, N_samples * w) for w in weights]
samples = reduce(vcat, randn(rng, n) * s .+ m for (n, s, m) in zip(Ns, stds, means))
sort!(samples)

##
fig = Figure();
ax = Axis(fig[1, 1], title = "Density and CDF");
density!(ax, samples, label = "Sample density")
e_cdf = (0:(N_samples - 1)) / (N_samples - 1)
lines!(ax, samples, e_cdf, label = "Emp. CDF", color = :red, linewidth = 3)
axislegend(position = :lt)
fig

##
gmm = MixtureModel([Normal(m, s) for (m, s) in zip(means, stds)], weights / sum(weights))
norm_dist = Normal()
map_from_dens = x -> invlogcdf(gmm, logcdf(norm_dist, x))
map_from_samp = x -> invlogcdf(norm_dist, logcdf(gmm, x))

##
xgrid = -5:0.05:5
fig = Figure();
ax = Axis(fig[1, 1], title = "Actual GMM Density and CDF");
band!(ax, xgrid, pdf.(gmm, xgrid), zeros(size(xgrid)), color = (:blue, 0.5), label = "PDF")
lines!(ax, xgrid, cdf.(gmm, xgrid), color = :red, linewidth = 3, label = "CDF")
sub_xgrid = xgrid[(xgrid .> -2) .& (xgrid .< 2)]
lines!(ax, sub_xgrid, map_from_dens.(sub_xgrid), label = "From Density", linewidth = 3)
lines!(ax, sub_xgrid, map_from_samp.(sub_xgrid), label = "From Sample", linewidth = 3)
axislegend(position = :lt)
fig

##
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
    func = OptimizationFunction(EvalLoss, grad = EvalLossGrad!)
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

function Distributions.gradlogpdf(d::AbstractMixtureModel, x::Float64)
    ps = probs(d)
    ds = components(d)
    inds = filter(i -> !iszero(ps[i]), eachindex(ps, ds))
    ws = Distributions.pweights(Distributions.softmax!([log(ps[i]) + logpdf(ds[i], x)
                                                        for i in inds]))
    g = mean(map(i -> gradlogpdf(ds[i], x), inds), ws)
    g
end

## From density
alg = LBFGS()
normdist = Normal()
max_degree = 10
mapLB, mapUB = -0.4, 1.6
left_tailwidth, right_tailwidth = 3, 4
centers = [quantile(norm_dist, (1:j) / (j + 1)) for j in 1:max_degree]
map_param = CreateSigmoidParam(; centers, mapLB, mapUB, left_tailwidth, right_tailwidth)
linmap = LinearMap(map_param, NumParams(map_param))
kl = KLDiv(x -> logpdf.(gmm, x), x -> gradlogpdf.(gmm, x))
# N_MC = 20_000
# qrule = MCQuad(randn(rng, N_MC))
N_quad = 200
qrule = BlackboxQuad(n -> gausshermite(n, normalize = true), N_quad)
TrainMap_PosConstraint(alg, linmap, qrule, kl)

evals = EvaluateMap(linmap, xgrid)
fig = Figure();
ax = Axis(fig[1, 1]);
lines!(ax, xgrid, evals, linewidth = 3, label = "Trained")
lines!(ax, xgrid, map_from_dens.(xgrid), linewidth = 3, label = "Exact")
axislegend()
fig

##
function InverseMap(umap::TransportMap,
        point::Real;
        lb::Real = -20,
        ub::Real = 20,
        maxiters = 3,)
    f = x -> EvaluateMap(umap, [x])[] - point
    tracks = Roots.Tracks()
    x_rough = find_zero(f, (lb, ub), Bisection(); maxiters, tracks)
    (tracks.convergence_flag != :not_converged) && return x_rough
    fp = x -> EvaluateMap(umap, [x], DerivativeFlags.InputGrad)[]
    x_fine = find_zero((f, fp), x_rough, Roots.Newton())
    x_fine
end

function TransportLogpdf_from_dens(umap::TransportMap,
        reference::Distribution,
        points::AbstractVector;
        inverse_kwargs...,)
    imap = point -> InverseMap(umap, point; inverse_kwargs...)
    map_points = imap.(points)
    eval_ref_logpdf = logpdf.(reference, map_points)
    eval_logdet = LogDeterminant(umap, map_points)
    eval_ref_logpdf - eval_logdet
end

##
test_set = randn(rng, 50_000)
eval_test = EvaluateMap(linmap, test_set)
transport_logpdf = TransportLogpdf_from_dens(linmap, norm_dist, xgrid)
fig = Figure();
ax = Axis(fig[1, 1]);
density!(ax, eval_test, label = "Test evaluations", color = (:red, 0.5))
lines!(ax, xgrid, exp.(transport_logpdf), label = "Transport PDF", linewidth = 3)
lines!(ax, xgrid, pdf.(gmm, xgrid), label = "Actual PDF", linewidth = 3, linestyle = :dash)
axislegend()
fig
