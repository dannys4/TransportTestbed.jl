include("gmm_base.jl")
using QuasiMonteCarlo, Random, Distributions, JLD2, ProgressMeter

# Note: will round N_pts up to the next power
function QMCNormal(rng::AbstractRNG, N_pts::Int, dim::Int = 1, SampleType::T = SobolSample, ScrambleType::U = OwenScramble) where {T,U}
    qmc_base = QuasiMonteCarlo.next_prime(dim)
    qmc_pad = ceil(Int, log(qmc_base, N_pts))
    N_pts_sample = qmc_base^(qmc_pad)
    unif_pts = QuasiMonteCarlo.sample(
        N_pts_sample, dim, SampleType(ScrambleType(qmc_base, qmc_pad, rng))
    )
    TransportTestbed.BoxMuller(unif_pts)
end

function FindQuadRule(rng::AbstractRNG, qtype::Symbol, N_pts::Int, description::Tuple)
    if qtype == :quadrature
        return BlackboxQuad(gaussprobhermite(), N_pts)
    elseif qtype == :montecarlo
        return MCQuad(randn(rng, N_pts))
    elseif qtype == :qmc
        normal_qsamples = QMCNormal(rng, N_pts)
        return MCQuad(normal_qsamples)
    elseif qtype == :hybrid_mc
        prop_mc = description[2]
        N_pts_mc = round(Int, prop_mc * N_pts)
        N_pts_quad = N_pts - N_pts_mc
        qrule_mc = MCQuad(randn(rng, N_pts_mc))
        qrule_gh = BlackboxQuad(gaussprobhermite(), N_pts_quad)
        return QuadPair(qrule_mc, qrule_gh; quad1_weight=prop_mc, quad2_weight=1 - prop_mc)
    elseif qtype == :hybrid_qmc
        prop_qmc = description[2]
        N_pts_qmc = round(Int, prop_qmc * N_pts)
        N_pts_quad = N_pts - N_pts_qmc
        uniform_samples = QuasiMonteCarlo.sample(N_pts_qmc, 0, 1, SobolSample())
        normal_dist = Normal()
        normal_samples = norminvcdf(uniform_samples[:])
        qrule_qmc = MCQuad(normal_samples)
        qrule_gh = BlackboxQuad(gaussprobhermite(), N_pts_quad)
        return QuadPair(
            qrule_qmc, qrule_gh; quad1_weight=prop_qmc, quad2_weight=1 - prop_qmc
        )
    else
        @error "Not implemented qtype $qtype"
    end
end

function CreateQuadRuleBench(rng::AbstractRNG, N_pts::Int, description::Tuple)
    qtype = description[1]
    !isa(qtype, Symbol) && throw(ArgumentError("First description arg should be symbol"))
    qrule = FindQuadRule(rng, qtype, N_pts, description)
    QuadRuleBench(N_pts, qrule, Ref{Float64}(), qtype)
end

function CreateAllQuadRuleBench(
    rng::AbstractRNG,
    num_points_range::AbstractVector{<:Int},
    qrule_desc::AbstractVector{<:Tuple},
)
    quadrules = [
        CreateQuadRuleBench(rng, N_pts, desc) for N_pts in num_points_range,
        desc in qrule_desc
    ]
    quadrules
end

function CreateRegBench(
    rng::AbstractRNG,
    N_pts::Int,
    qrule_desc::Tuple,
    l2_reg::Float64,
    sob_reg::Float64
)
    qtype = qrule_desc[1]
    qrule = FindQuadRule(rng, qtype, N_pts, qrule_desc)
    RegBench(N_pts, qrule, Ref{Float64}(), l2_reg, sob_reg)
end

function CreateAllRegBench_converge(
    rng::AbstractRNG,
    num_pt_range::AbstractVector{<:Int},
    qrule_desc::Tuple,
    regularizations::AbstractVector{<:Tuple{Float64,Float64}}
)
    [CreateRegBench(rng, N_pts, qrule_desc, l2_reg, sob_reg) for N_pts in num_pt_range, (l2_reg, sob_reg) in regularizations]
end

function CreateAllRegBench_tuneReg(
    rng::AbstractRNG,
    N_pts::Int,
    qrule_desc::Tuple,
    l2_regs::AbstractVector{Float64},
    sob_regs::AbstractVector{Float64}
)
    [CreateRegBench(rng, N_pts, qrule_desc, l2_reg, sob_reg) for l2_reg in l2_regs, sob_reg in sob_regs]
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

function AssessError!(
    error_ref::Ref{Float64},
    qrule,
    target_dist::Distribution,
    error_pts;
    loss=:kl,
    alg=LBFGS(),
    L2_reg=0.,
    sob_reg=0.
)
    loss != :kl && throw(ArgumentError("Only accept loss as :kl, given $loss"))
    kl_loss = KLDiv(x -> logpdf.(target_dist, x), x -> gradlogpdf.(target_dist, x))
    loss_obj = CombinedLoss(kl_loss, weight_param_reg = L2_reg, weight_sobolev_reg = sob_reg)
    umap = DefaultMap()
    TrainMap_PosConstraint!(alg, umap, qrule, loss_obj, false)
    error_ref[] = KolmogorovSmirnov(umap, target_dist, error_pts)
end

function CreateQuadRuleErrors(
    rng::AbstractRNG, target_dist::Distribution, N_error_pts::Int=10_000, n_pt_range = nothing
)
    quad_descs = [
        (:quadrature,),
        (:montecarlo,),
        (:qmc,),
        (:hybrid_mc, 0.9),
        (:hybrid_mc, 0.75),
        (:hybrid_mc, 0.5),
        (:hybrid_mc, 0.25),
        (:hybrid_mc, 0.1),
        (:hybrid_qmc, 0.9),
        (:hybrid_qmc, 0.75),
        (:hybrid_qmc, 0.5),
        (:hybrid_qmc, 0.25),
        (:hybrid_qmc, 0.1),
    ]
    isnothing(n_pt_range) && (n_pt_range = round.(Int, 10 .^ (1:0.25:3)))
    quadrules = CreateAllQuadRuleBench(rng, n_pt_range, quad_descs)
    error_pts = randn(rng, N_error_pts)
    p = Progress(length(quadrules); dt=1, desc="Assessing qrule error...")
    Threads.@threads for j in eachindex(quadrules)
        qrule_bench = quadrules[j]
        AssessError!(qrule_bench.error, qrule_bench.qrule, target_dist, error_pts)
        jldsave(joinpath(dirname(@__DIR__), "data", "tmp", "quadrule_error_$(j).jld2");qrule_bench)
        next!(p)
    end
    finish!(p)
    jldsave(joinpath(dirname(@__DIR__), "data", "quadrule_error.jld2"))
    quadrules
end

function CreateRegularizationErrors(
    rng::AbstractRNG, target_dist::Distribution; N_pt_tune::Int = 60, N_error_pts::Int=10_000, num_pt_range = nothing, bench_conv::Bool = true, bench_tune::Bool = true
)
    quad_desc = (:quadrature,)
    isnothing(num_pt_range) && (num_pt_range = round.(Int, 10 .^ (1:0.2:2)))
    frac_inc = 3 # Increment by a multiple of 10^{1/frac_inc}
    reg_log_min = -5 # start at 10^{reg_log_min}
    reg_log_max = -1 # end at 10^{reg_log_max}
    reg_range = 10 .^ ((reg_log_min*frac_inc:reg_log_max*frac_inc)/frac_inc)
    reg_conv = vcat([(0.,0.)], [(r,0.) for r in reg_range], [(0.,r) for r in reg_range])
    reg_bench_conv = CreateAllRegBench_converge(rng, num_pt_range, quad_desc, reg_conv)
    reg_bench_tune = CreateAllRegBench_tuneReg(rng, N_pt_tune, quad_desc, reg_range, reg_range)
    error_pts = randn(rng, N_error_pts)
    tmp_save_path = (kind,j)->joinpath(dirname(@__DIR__), "data", "tmp", "reg_$(kind)_$(j).jld2")
    total_prog_conv = sum(reg_bench.N for reg_bench in reg_bench_conv)
    total_prog_tune = sum(reg_bench.N for reg_bench in reg_bench_tune)
    if bench_conv
        p = Progress(total_prog_conv; dt=1, desc="Assessing regularization convergence...")
        Threads.@threads for j in eachindex(reg_bench_conv)
            reg_bench = reg_bench_conv[j]
            AssessError!(reg_bench.error, reg_bench.qrule, target_dist, error_pts; L2_reg=reg_bench.l2_reg, sob_reg=reg_bench.sob_reg)
            jldsave(tmp_save_path("conv",j);reg_bench)
            next!(p, step=reg_bench.N)
        end
        finish!(p)
        jldsave(joinpath(dirname(@__DIR__), "data", "reg_conv_all.jld2"); reg_bench_conv)
    else
        reg_bench_conv = nothing
    end
    if bench_tune
        p = Progress(total_prog_tune; dt=1, desc="Assessing regularization tuning...")
        Threads.@threads for j in eachindex(reg_bench_tune)
            reg_bench = reg_bench_tune[j]
            AssessError!(reg_bench.error, reg_bench.qrule, target_dist, error_pts; L2_reg=reg_bench.l2_reg, sob_reg=reg_bench.sob_reg)
            jldsave(tmp_save_path("tune",j);reg_bench)
            next!(p, step=reg_bench.N)
        end
        finish!(p)
        jldsave(joinpath(dirname(@__DIR__), "data", "reg_tune_all.jld2"); reg_bench_tune)
    else
        reg_bench_tune = nothing
    end
    reg_bench_conv, reg_bench_tune
end

##
# quadrules = CreateQuadRuleErrors(rng, gmm)
reg_bench_conv, reg_bench_tune = CreateRegularizationErrors(rng, gmm)