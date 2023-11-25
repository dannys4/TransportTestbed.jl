include("gmm_base.jl")
using GLMakie, QuasiMonteCarlo, Random, Distributions, JLD2

function CreateQuadRuleBench(rng::AbstractRNG, N_pts::Int, description::Tuple)
    qtype = description[1]
    !isa(qtype, Symbol) && throw(ArgumentError("First description arg should be symbol"))
    qrule = nothing
    if qtype == :quadrature
        qrule = BlackboxQuad(gaussprobhermite(), N_pts)
    elseif qtype == :montecarlo
        qrule = MCQuad(randn(rng, N_pts))
    elseif qtype == :qmc
        uniform_samples = QuasiMonteCarlo.sample(N_pts, 0, 1, SobolSample())
        normal_dist = Normal()
        normal_samples = quantile(normal_dist, uniform_samples[:])
        qrule = MCQuad(normal_samples)
    elseif qtype == :hybrid_mc
        prop_mc = description[2]
        N_pts_mc = round(Int, prop_mc*N_pts)
        N_pts_quad = N_pts-N_pts_mc
        qrule_mc = MCQuad(randn(rng, N_pts_mc))
        qrule_gh = BlackboxQuad(gaussprobhermite(), N_pts_quad)
        qrule = QuadPair(qrule_mc, qrule_gh, quad1_weight=prop_mc, quad2_weight=1-prop_mc)
    elseif qtype == :hybrid_qmc
        prop_qmc = description[2]
        N_pts_qmc = round(Int, prop_qmc*N_pts)
        N_pts_quad = N_pts-N_pts_qmc
        uniform_samples = QuasiMonteCarlo.sample(N_pts_qmc, 0, 1, SobolSample())
        normal_dist = Normal()
        normal_samples = quantile(normal_dist, uniform_samples[:])
        qrule_qmc = MCQuad(normal_samples)
        qrule_gh = BlackboxQuad(gaussprobhermite(), N_pts_quad)
        qrule = QuadPair(qrule_qmc, qrule_gh, quad1_weight=prop_qmc, quad2_weight=1-prop_qmc)
    else
        @error "Not implemented qtype $qtype"
    end
    QuadRuleBench(N_pts, qrule, Ref{Float64}(), qtype)
end

function MakeQuadRules(
    rng::AbstractRNG,
    num_points_range::AbstractVector{<:Int},
    qrule_desc::AbstractVector{<:Tuple},
)
    quadrules = [CreateQuadRuleBench(rng, N_pts, desc) for N_pts in num_points_range, desc in qrule_desc]
    quadrules
end

function KolmogorovSmirnov(umap::TransportMap, target_dist::Distribution, eval_pts::AbstractVector)
    N_pts = length(eval_pts)
    map_eval = EvaluateMap(umap, eval_pts)
    sort!(map_eval)
    e_cdf = collect((1:N_pts)/N_pts)
    true_cdf = cdf(target_dist, map_eval)
    maximum(abs.(e_cdf - true_cdf))
end

function AssessError!(error_ref::Ref{Float64}, qrule, target_dist::Distribution, error_pts, loss = :kl, alg = LBFGS())
    loss != :kl && throw(ArgumentError("Only accept loss as :kl, given $loss"))
    loss_obj = KLDiv(x->logpdf.(target_dist, x), x->gradlogpdf.(target_dist, x))
    umap = DefaultMap()
    TrainMap_PosConstraint!(alg, umap, qrule, loss_obj, false)
    error_ref[] = KolmogorovSmirnov(umap, target_dist, error_pts)
end

function CreateQuadRuleErrors(rng::AbstractRNG, target_dist::Distribution, N_error_pts::Int = 10_000)
    quad_descs = [
        (:quadrature,), (:montecarlo,), (:qmc,),
        (:hybrid_mc,0.9), (:hybrid_mc,0.75), (:hybrid_mc,0.5), (:hybrid_mc,0.25), (:hybrid_mc,0.1),
        (:hybrid_qmc,0.9), (:hybrid_qmc,0.75), (:hybrid_qmc,0.5), (:hybrid_qmc,0.25), (:hybrid_qmc,0.1),
    ]
    n_pt_range = round.(Int, 10 .^ (1:0.25:3))
    quadrules = MakeQuadRules(rng, n_pt_range, quad_descs)
    error_pts = randn(rng, N_error_pts)
    for qrule_bench in quadrules
        AssessError!(qrule_bench.error, qrule_bench.qrule, target_dist, error_pts)
    end
    quadrules
end

quadrules = CreateQuadRuleErrors(rng, gmm)
jldsave(joinpath(@__DIR__, "data/quadrule_error.jld2"); quadrules)