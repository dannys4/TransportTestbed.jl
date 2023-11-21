using TransportTestbed, GLMakie, Random, Distributions, Optimization, OptimizationOptimJL, FastGaussQuadrature
rng = Xoshiro(392204)
N_samples = 20_000
stds = [1.5, 0.5]
means = [-0.5,0.5]
weights = [0.5, 0.5]

##
Ns = [round(Int, N_samples*w) for w in weights]
samples = reduce(vcat, randn(rng, n)*s .+ m for (n,s,m) in zip(Ns, stds, means))
sort!(samples)

##
fig = Figure(); ax = Axis(fig[1,1], title="Density and CDF")
density!(ax, samples, label="Sample density")
e_cdf = (0:N_samples-1)/(N_samples-1)
lines!(ax, samples, e_cdf, label="Emp. CDF", color=:red, linewidth=3)
axislegend(position=:lt)
fig

##
gmm = MixtureModel([Normal(m,s) for (m,s) in zip(means,stds)], weights/sum(weights))
norm_dist = Normal()
map_from_dens = x -> invlogcdf(gmm, logcdf(norm_dist, x))
map_from_samp = x -> invlogcdf(norm_dist, logcdf(gmm, x))

##
xgrid = -5:0.05:5
fig = Figure(); ax = Axis(fig[1,1], title="Actual GMM Density and CDF")
band!(ax, xgrid, pdf.(gmm, xgrid), zeros(size(xgrid)), color=(:blue,0.5), label="PDF")
lines!(ax, xgrid, cdf.(gmm, xgrid), color=:red, linewidth=3, label="CDF")
sub_xgrid = xgrid[(xgrid .> -2) .& (xgrid .< 2)]
lines!(ax, sub_xgrid, map_from_dens.(sub_xgrid), label="From Density", linewidth=3)
lines!(ax, sub_xgrid, map_from_samp.(sub_xgrid), label="From Sample", linewidth=3)
axislegend(position=:lt)
fig

##
sp = TransportTestbed.SoftPlus
softplus_eval = (t::Float64) -> TransportTestbed.Evaluate(sp, t)
softplus_grad = (t::Float64) -> TransportTestbed.Derivative(sp, t)
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
    func = OptimizationFunction(EvalLoss, grad=EvalLossGrad!)
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
    ws = Distributions.pweights(Distributions.softmax!([log(ps[i]) + logpdf(ds[i], x) for i in inds]))
    g = mean(map(i -> gradlogpdf(ds[i], x), inds), ws)
    g
end

## From density
alg = LBFGS()
normdist = Normal()
max_degree = 10
mapLB, mapUB = -0.4, 1.6
bound_dist = mapUB-mapLB
centers = [quantile(norm_dist, (1:j)/(j+1)) for j in 1:max_degree]
map_param = CreateSigmoidParam(;centers, mapLB, mapUB)
linmap = LinearMap(map_param, NumParams(map_param))
kl = KLDiv(x->logpdf.(gmm,x), x->gradlogpdf.(gmm, x))
# N_MC = 20_000
# qrule = MCQuad(randn(rng, N_MC))
N_quad = 200
qrule = BlackboxQuad(n->gausshermite(n,normalize=true), N_quad)
TrainMap_PosConstraint(alg, linmap, qrule, kl)

evals = EvaluateMap(linmap, xgrid)
fig = Figure(); ax = Axis(fig[1,1])
lines!(ax, xgrid, evals, linewidth=3, label="Trained")
lines!(ax, xgrid, map_from_dens.(xgrid), linewidth=3, label="Exact")
axislegend()
fig

##
test_set = randn(rng, 50_000)
eval_test = EvaluateMap(linmap, test_set)
fig = Figure(); ax = Axis(fig[1,1])
density!(ax, eval_test, label="Test evaluations")
lines!(ax, xgrid, pdf.(gmm, xgrid), label="Actual PDF")
axislegend()
fig

##
fd_delta = 1e-5
left_slope=(map_from_dens(-5+fd_delta)-map_from_dens(-5))/fd_delta
right_slope=(map_from_dens(5+fd_delta)-map_from_dens(5))/fd_delta
left_lin, right_lin = -0.88, 2.25
left_width, right_width = 3,4
offset = -1.45
tails = x->offset-(left_slope/left_width)*softplus_eval(-left_width*(x-left_lin)) + (right_slope/right_width)*softplus_eval(right_width*(x-right_lin))
fig, ax, _ = lines(xgrid, map_from_dens.(xgrid),label="Real", linewidth=3)
lines!(ax, xgrid, tails.(xgrid), label="Fake", linewidth=3)
vlines!(ax, 1.5)
axislegend()
fig

##
xgrid = -6:0.01:6
fig = Figure(); ax = Axis(fig[1,1])
diff_fcn = x->map_from_dens(x) - tails(x)
sigmoid = x-> x < 0 ? exp(x)/(1+exp(x)) : 1/(1+exp(-x))
width1, amp1, offset1 = 0.5, 1.4, -0.8
sig1 = x->amp1*sigmoid((x-offset1)/width1)
width2, amp2, offset2 = 0.2, 0.48, -0.8
sig2 = x->amp2*sigmoid((x-offset2)/width2)
width3, amp3, offset3 = 0.5,2.,1.65
sig3 = x->amp3*sigmoid((x-offset3)/width3)
width4, amp4, offset4 = 0.25,0.15,0.2
sig4 = x->amp4*sigmoid((x-offset4)/width4)
width5, amp5, offset5 = 0.1,0.0,0.19
sig5 = x->amp5*sigmoid((x-offset5)/width5)
lines!(ax, xgrid, diff_fcn.(xgrid), label="diff", linewidth=3)
lines!(ax, xgrid, sig1.(xgrid), label="fake", linewidth=3)
lines!(ax, xgrid, diff_fcn.(xgrid)-sig1.(xgrid), label="diff2", linewidth=3)
lines!(ax, xgrid, sig2.(xgrid), label="fake2", linewidth=3)
lines!(ax, xgrid, diff_fcn.(xgrid)-sig1.(xgrid)-sig2.(xgrid), label="diff3", linewidth=3)
lines!(ax, xgrid, sig3.(xgrid), label="fake3", linewidth=3)
lines!(ax, xgrid, diff_fcn.(xgrid)-sig1.(xgrid)-sig2.(xgrid)-sig3.(xgrid), label="diff4", linewidth=3)
lines!(ax, xgrid, sig4.(xgrid), label="fake4", linewidth=3)
lines!(ax, xgrid, diff_fcn.(xgrid)-sig1.(xgrid)-sig2.(xgrid)-sig3.(xgrid)-sig4.(xgrid), label="diff5", linewidth=3)
# lines!(ax, xgrid, sig5.(xgrid), label="fake5", linewidth=3)
# lines!(ax, xgrid, diff_fcn.(xgrid)-sig1.(xgrid)-sig2.(xgrid)-sig3.(xgrid)-sig4.(xgrid)-sig5.(xgrid), label="diff6", linewidth=3)
axislegend(position=:lt)
fig

##
approx_map = x->sum(f(x) for f in [tails, sig1, sig2, sig3, sig4, sig5])
fig = Figure(); ax=Axis(fig[1,1])
lines!(ax, xgrid, map_from_dens.(xgrid), label="True", linewidth=3)
lines!(ax, xgrid, approx_map.(xgrid), label="Approx", linewidth=3, linestyle=:dash)
fig

##
fig = Figure(); ax = Axis(fig[1,1])
band!(ax, xgrid, zeros(length(xgrid)), pdf.(gmm, xgrid), color=(:blue, 0.25))
vlines!(ax, [offset1, offset2, offset3, offset4], label="Centers")
vlines!(ax, [left_lin, right_lin], label="Tails", linewidth=5)
vlines!(ax, means, label="Modes", linewidth=5, linestyle=:dash)
axislegend()
fig

##
fig, ax, _ = lines(xgrid, map_from_dens.(xgrid), label="Exact", linewidth=4)
vlines!(ax, [left_lin, right_lin], label="Tails", linewidth=4)
fig

##
diff_normal = logpdf.(gmm, map_from_dens.(xgrid)) - logpdf.(normdist, xgrid)
fig, ax, _ = band(xgrid, min.(diff_normal, 0), max.(diff_normal,0))
vlines!(ax, [left_lin, 2.25], label="Tails", linewidth=5)
vlines!(ax, [offset1, offset2, offset3, offset4], linewidth=3)
fig

##
d_sigmoid = x->sigmoid(x)*(1-sigmoid(x))
d2_sigmoid = x->sigmoid(x)*(1-sigmoid(x))*(1-2sigmoid(x))
fig, ax, _ = lines(xgrid, sigmoid.(xgrid))
lines!(ax, xgrid, d_sigmoid.(xgrid))
lines!(ax, xgrid, d2_sigmoid.(xgrid))
fig

##
fig, ax, _ = lines(xgrid, gradlogpdf.(gmm, xgrid))
vlines!(ax, [left_lin, right_lin])
vlines!(ax, [offset1, offset2, offset3, offset4])
fig