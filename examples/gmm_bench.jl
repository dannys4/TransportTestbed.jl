using GLMakie
include("gmm_base.jl")

struct QuadRulePlotArgs
    name::String
    color::Symbol
    marker::Symbol
end
struct QuadRuleBench
    N::Int
    qrule::QuadRule
    plot_args::QuadRulePlotArgs
    error::Ref{Float64}
end

COLORS = Makie.wong_colors()

HYBRID_PLOT_DICT = Dict(
    (0.90, 0.10) => nothing,
    (0.75, 0.25) => nothing,
    (0.50, 0.50) => nothing,
    (0.25, 0.75) => nothing,
    (0.10, 0.90) => nothing,
)

PLOT_DICT = Dict(
    :quadrature => QuadRulePlotArgs("Quadrature", COLORS[1], :rect),
    :montecarlo => QuadRulePlotArgs("Monte Carlo", COLORS[2], :rect),
    :qmc => QuadRulePlotArgs("QMC", COLORS[3], :rect),
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
    range_len = length(num_points_range)
    num_qrules = length(qrule_desc)
    return quadrules = Matrix{QuadRuleBench}(undef, range_len, num_qrules)
end
