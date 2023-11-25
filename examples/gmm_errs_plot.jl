include("gmm_base.jl")

using DataFrames, JLD2, GLMakie

COLORS = Makie.wong_colors()

# prop_mc => args
HYBRID_MC_PLOT_DICT = Dict(
    0.10 => QuadRulePlotArgs("Hybrid MC/Quad, 10%", COLORS[1], :rect),
    0.25 => QuadRulePlotArgs("Hybrid MC/Quad, 25%", COLORS[2], :circle),
    0.50 => QuadRulePlotArgs("Hybrid MC/Quad, 50%", COLORS[3], :utriangle),
    0.75 => QuadRulePlotArgs("Hybrid MC/Quad, 75%", COLORS[4], :star4),
    0.90 => QuadRulePlotArgs("Hybrid MC/Quad, 90%", COLORS[5], :diamond),
)

# prop_qmc => args
HYBRID_QMC_PLOT_DICT = Dict(
    0.10 => QuadRulePlotArgs("Hybrid QMC/Quad, 10%", COLORS[1], :rect),
    0.25 => QuadRulePlotArgs("Hybrid QMC/Quad, 25%", COLORS[2], :circle),
    0.50 => QuadRulePlotArgs("Hybrid QMC/Quad, 50%", COLORS[3], :utriangle),
    0.75 => QuadRulePlotArgs("Hybrid QMC/Quad, 75%", COLORS[4], :star4),
    0.90 => QuadRulePlotArgs("Hybrid QMC/Quad, 90%", COLORS[5], :diamond),
)

PLOT_DICT = Dict(
    :quadrature => QuadRulePlotArgs("Gauss-Hermite", COLORS[1], :rect),
    :montecarlo => QuadRulePlotArgs("Monte Carlo", COLORS[2], :rect),
    :qmc => QuadRulePlotArgs("Mapped Sobol QMC", COLORS[3], :rect),
    :hybrid_mc => nothing,
    :hybrid_qmc => nothing
)


qrules = jldopen(joinpath(@__DIR__, "data", "quadrule_error.jld2"))["quadrules"]

##
quad_conv = qrules[:,1]
mc_conv = qrules[:,2]
qmc_conv = qrules[:,3]
fig = Figure()
ax = Axis(fig[1,1], xscale=log10, yscale=log10, xlabel="N", ylabel="KS-statistic", title="Convergence of 'pure' methods, 1D")
for conv in [quad_conv, mc_conv, qmc_conv]
    plot_args = PLOT_DICT[conv[1].name]
    Ns = [c.N for c in conv]
    errs = [c.error[] for c in conv]
    scatterlines!(ax, Ns, errs, label=plot_args.name, linewidth=3, marker=plot_args.marker)
end
axislegend()
save(joinpath(@__DIR__, "figs", "pure_conv.png"), fig)