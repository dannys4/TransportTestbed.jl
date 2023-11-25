include("gmm_base.jl")
##
N_samples = 20_000
fig = Figure()
ax = Axis(fig[1, 1]; title="Density and CDF")
samples = rand(rng, gmm, N_samples)
sort!(samples)
density!(ax, samples; label="Sample density")
e_cdf = (0:(N_samples - 1)) / (N_samples - 1)
lines!(ax, samples, e_cdf; label="Emp. CDF", color=:red, linewidth=3)
axislegend(; position=:lt)
fig

##
xgrid = -5:0.05:5
fig = Figure();
ax = Axis(fig[1, 1]; title="Actual GMM Density and CDF");
band!(ax, xgrid, pdf.(gmm, xgrid), zeros(size(xgrid)); color=(:blue, 0.5), label="PDF")
lines!(ax, xgrid, cdf.(gmm, xgrid); color=:red, linewidth=3, label="CDF")
sub_xgrid = xgrid[(xgrid .> -2) .& (xgrid .< 2)]
lines!(ax, sub_xgrid, map_from_dens.(sub_xgrid); label="From Density", linewidth=3)
lines!(ax, sub_xgrid, map_from_samp.(sub_xgrid); label="From Sample", linewidth=3)
axislegend(; position=:lt)
fig

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
N_quad, qrule = 200, nothing
# Change this as needed
qrule_choice = :quadrature
if qrule_choice == :quadrature
    qrule = BlackboxQuad(n -> gausshermite(n; normalize=true), N_quad)
elseif qrule_choice == :montecarlo
    qrule = MCQuad(randn(rng, N_quad))
end
TrainMap_PosConstraint(alg, linmap, qrule, kl)

evals = EvaluateMap(linmap, xgrid)
fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, xgrid, evals; linewidth=3, label="Trained")
lines!(ax, xgrid, map_from_dens.(xgrid); linewidth=3, label="Exact")
axislegend()
fig

##
N_test_set = 50_000
test_set = randn(rng, N_test_set)
eval_test = EvaluateMap(linmap, test_set)
transport_logpdf = TransportLogpdf_from_dens(linmap, norm_dist, xgrid)
fig = Figure()
ax = Axis(fig[1, 1])
density!(ax, eval_test; label="Test evaluations", color=(:red, 0.5))
lines!(ax, xgrid, exp.(transport_logpdf); label="Transport PDF", linewidth=3)
lines!(ax, xgrid, pdf.(gmm, xgrid); label="Actual PDF", linewidth=3, linestyle=:dash)
axislegend()
fig
