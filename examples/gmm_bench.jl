using GLMakie, QuasiMonteCarlo, Random, Distributions
include("gmm_base.jl")

struct QuadRulePlotArgs
    name::String
    color
    marker::Symbol
end

struct QuadRuleBench
    N::Int
    qrule::TransportTestbed.QuadRule
    plot_args::QuadRulePlotArgs
    error::Ref{Float64}
end

COLORS = Makie.wong_colors()

# prop_mc => args
HYBRID_MC_PLOT_DICT = Dict(
    0.10 => QuadRulePlotArgs("Hybric MC/Quad, 10%", COLORS[1], :rect),
    0.25 => QuadRulePlotArgs("Hybric MC/Quad, 25%", COLORS[2], :circle),
    0.50 => QuadRulePlotArgs("Hybric MC/Quad, 50%", COLORS[3], :utriangle),
    0.75 => QuadRulePlotArgs("Hybric MC/Quad, 75%", COLORS[4], :star4),
    0.90 => QuadRulePlotArgs("Hybric MC/Quad, 90%", COLORS[5], :diamond),
)

# prop_qmc => args
HYBRID_QMC_PLOT_DICT = Dict(
    0.10 => QuadRulePlotArgs("Hybric QMC/Quad, 10%", COLORS[1], :rect),
    0.25 => QuadRulePlotArgs("Hybric QMC/Quad, 25%", COLORS[2], :circle),
    0.50 => QuadRulePlotArgs("Hybric QMC/Quad, 50%", COLORS[3], :utriangle),
    0.75 => QuadRulePlotArgs("Hybric QMC/Quad, 75%", COLORS[4], :star4),
    0.90 => QuadRulePlotArgs("Hybric QMC/Quad, 90%", COLORS[5], :diamond),
)

PLOT_DICT = Dict(
    :quadrature => QuadRulePlotArgs("Quadrature", COLORS[1], :rect),
    :montecarlo => QuadRulePlotArgs("Monte Carlo", COLORS[2], :rect),
    :qmc => QuadRulePlotArgs("QMC", COLORS[3], :rect),
    :hybrid_mc => nothing,
    :hybrid_qmc => nothing
)

function CreateQuadRuleBench(rng::AbstractRNG, N_pts::Int, description::Tuple)
    qtype = description[1]
    !isa(qtype, Symbol) && throw(ArgumentError("First description arg should be symbol"))
    plot_args = PLOT_DICT[qtype]
    qrule = nothing
    if qtype == :quadrature
        qrule = BlackboxQuad(gaussprobhermite, N_pts)
    elseif qtype == :montecarlo
        qrule = MCQuad(randn(rng, N_pts))
    elseif qtype == :qmc
        uniform_samples = QuasiMonteCarlo.sample(N_pts, 0, 1, SobolSample())
        normal_dist = Normal()
        normal_samples = quantile(normal_dist, uniform_samples[:])
        qrule = MCQuad(normal_samples)
    elseif qtype == :hybrid_mc
        prop_mc = description[2]
        plot_args = HYBRID_MC_PLOT_DICT[prop_mc]
        N_pts_mc = round(Int, prop_mc*N_pts)
        N_pts_quad = N_pts-N_pts_mc
        qrule_mc = MCQuad(randn(rng, N_pts_mc))
        qrule_gh = BlackboxQuad(gaussprobhermite, N_pts_quad)
        qrule = QuadPair(qrule_mc, qrule_gh, quad1_weight=prop_mc, quad2_weight=1-prop_mc)
    elseif qtype == :hybrid_qmc
        prop_qmc = description[2]
        plot_args = HYBRID_QMC_PLOT_DICT[prop_qmc]
        N_pts_qmc = round(Int, prop_qmc*N_pts)
        N_pts_quad = N_pts-N_pts_qmc
        uniform_samples = QuasiMonteCarlo.sample(N_pts_qmc, 0, 1, SobolSample())
        normal_dist = Normal()
        normal_samples = quantile(normal_dist, uniform_samples[:])
        qrule_qmc = MCQuad(normal_samples)
        qrule_gh = BlackboxQuad(gaussprobhermite, N_pts_quad)
        qrule = QuadPair(qrule_qmc, qrule_gh, quad1_weight=prop_qmc, quad2_weight=1-prop_qmc)
    else
        @error "Not implemented qtype $qtype"
    end
    return QuadRuleBench(N_pts, qrule, plot_args, Ref{Float64}())
end

function MakeQuadRules(
    rng::AbstractRNG,
    num_points_range::AbstractVector{<:Int},
    qrule_desc::AbstractVector{<:Tuple},
)
    quadrules = [CreateQuadRuleBench(rng, N_pts, desc) for N_pts in num_points_range, desc in qrule_desc]
    quadrules
end

quad_descs = [
    (:quadrature,), (:montecarlo,), (:qmc,),
    (:hybrid_mc,0.9), (:hybrid_mc,0.75), (:hybrid_mc,0.5), (:hybrid_mc,0.25), (:hybrid_mc,0.1),
    (:hybrid_qmc,0.9), (:hybrid_qmc,0.75), (:hybrid_qmc,0.5), (:hybrid_qmc,0.25), (:hybrid_qmc,0.1),
]
n_pt_range = 10 .^ (1:0.25:3)
quadrules = MakeQuadRules(rng, 10:10:100, quad_descs)
