include("gmm_base.jl")

using DataFrames, JLD2

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
    :quadrature => QuadRulePlotArgs("Quadrature", COLORS[1], :rect),
    :montecarlo => QuadRulePlotArgs("Monte Carlo", COLORS[2], :rect),
    :qmc => QuadRulePlotArgs("QMC", COLORS[3], :rect),
    :hybrid_mc => nothing,
    :hybrid_qmc => nothing
)



qrules_jld2 = jldopen(joinpath(@__DIR__, "data", "quadrule_error.jld2"))["quadrules"]
