include("gmm_base.jl")

using DataFrames, JLD2, GLMakie

COLORS = Makie.wong_colors()

# prop_mc or prop_qmc => args
HYBRID_QUAD_PLOT_DICT = Dict(
    0.10 => BenchPlotArgs("10%", COLORS[1], :rect),
    0.25 => BenchPlotArgs("25%", COLORS[2], :circle),
    0.50 => BenchPlotArgs("50%", COLORS[3], :utriangle),
    0.75 => BenchPlotArgs("75%", COLORS[4], :star4),
    0.90 => BenchPlotArgs("90%", COLORS[5], :diamond),
)

QUAD_PLOT_DICT = Dict(
    :quadrature => BenchPlotArgs("Gauss-Hermite", COLORS[1], :rect),
    :montecarlo => BenchPlotArgs("Monte Carlo", COLORS[2], :circle),
    :qmc => BenchPlotArgs("Mapped Sobol QMC", COLORS[3], :utriangle),
    :hybrid_mc => nothing,
    :hybrid_qmc => nothing,
)

REG_CONV_PLOT_DICT = Dict(
    :l2  => BenchPlotArgs(L"L^2", COLORS[1], :rect),
    :sob => BenchPlotArgs(L"W^{1,2}", COLORS[2], :circle),
    :ref => BenchPlotArgs(L"$ $None", COLORS[3], :utriangle)
)

module QruleIndex
@enum __QruleIndex Quadrature = 1 MonteCarlo = 2 QMC = 3 HybridMCStart = 4 HybridMCEnd = 8 HybridQMCStart =
    9 HybridQMCEnd = 13
end

function CreatePureQuadConvergencePlot(qrules)
    quad_conv = qrules[:, Int(QruleIndex.Quadrature)]
    mc_conv = qrules[:, Int(QruleIndex.MonteCarlo)]
    qmc_conv = qrules[:, Int(QruleIndex.QMC)]
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xscale=log10,
        yscale=log10,
        xlabel="N",
        ylabel="KS-statistic",
        title="Convergence of 'pure' methods, 1D",
    )
    for conv in [quad_conv, mc_conv, qmc_conv]
        plot_args = QUAD_PLOT_DICT[conv[1].name]
        Ns = [c.N for c in conv]
        errs = [c.error[] for c in conv]
        scatterlines!(
            ax, Ns, errs; label=plot_args.name, linewidth=3, marker=plot_args.marker
        )
    end
    axislegend()
    save(joinpath(@__DIR__, "figs", "pure_conv.png"), fig)
    fig
end

function CreateHybridQuadConvergencePlot(qrules, which_mc)
    start_idx = end_idx = 0
    title_suff = nothing
    if which_mc == :hybrid_mc
        start_idx = Int(QruleIndex.HybridMCStart)
        end_idx = Int(QruleIndex.HybridMCEnd)
        title_suff = "MC"
    elseif which_mc == :hybrid_qmc
        start_idx = Int(QruleIndex.HybridQMCStart)
        end_idx = Int(QruleIndex.HybridQMCEnd)
        title_suff = "QMC"
    else
        throw(ArgumentError("Given invalid argument which_mc=$which_mc"))
    end
    qrules_sub = qrules[:, start_idx:end_idx]
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title="Convergence of Hybrid method, $title_suff",
        xlabel="N",
        ylabel="KS-statistic",
        xscale=log10,
        yscale=log10,
    )
    plot_lamb = (Ns, errs, plot_args;kwargs...) -> scatterlines!(ax, Ns, errs, label=plot_args.name, color=plot_args.color, marker=plot_args.marker, linewidth=3;kwargs...)
    for j in axes(qrules_sub, 2)
        col = qrules_sub[:,j]
        prop_mc = round(col[1].qrule.quad1_weight, digits=2)
        plot_args = HYBRID_QUAD_PLOT_DICT[prop_mc]
        Ns = [c.N for c in col]
        errs = [c.error[] for c in col]
        plot_lamb(Ns, errs, plot_args)
    end
    col = qrules[:,Int(QruleIndex.Quadrature)]
    plot_args = QUAD_PLOT_DICT[:quadrature]
    Ns = [c.N for c in col]
    errs = [c.error[] for c in col]
    plot_lamb(Ns, errs, plot_args, linestyle=:dash)
    axislegend()
    save(joinpath(@__DIR__, "figs", "hybrid_conv_$(title_suff).png"), fig)
    fig
end
##
function CreateMarginalRegConvergencePlot(reg_bench_conv)
    fig = Figure(size=(1200,800))
    ref_conv = filter(r->r.l2_reg == 0 && r.sob_reg == 0, reg_bench_conv)
    N_vals = map(r->r.N, ref_conv)
    num_cols = round(Int, 3*sqrt(length(N_vals)/6)) # Should be 3L x 2L roughly
    l2_plot_args = REG_CONV_PLOT_DICT[:l2]
    sob_plot_args = REG_CONV_PLOT_DICT[:sob]
    for j in eachindex(N_vals)
        row, col = ((j-1) ÷ num_cols) + 1, ((j-1) % num_cols) + 1
        N = N_vals[j]
        ax = Axis(fig[row,col], title="Convergence of regularization, N=$N", xlabel=L"\lambda", ylabel="KS-statistic", xscale=log10, yscale=log10)
        ref_err = ref_conv[j].error[]
        l2_conv = filter(r->r.l2_reg != 0 && r.sob_reg == 0 && r.N == N, reg_bench_conv)
        sob_conv = filter(r->r.l2_reg == 0 && r.sob_reg != 0 && r.N == N, reg_bench_conv)
        l2_reg_range = map(r->r.l2_reg, l2_conv)
        ref_err_repeat = fill(ref_conv[j].error[], length(l2_conv))
        plot_lamb = (regs, errs, plot_args; kwargs...) -> scatterlines!(ax, regs, errs, label=plot_args.name, color=plot_args.color, marker=plot_args.marker, linewidth=3, markersize=15;kwargs...)
        plot_lamb(l2_reg_range, map(r->r.error[], l2_conv_min), l2_plot_args)
        plot_lamb(map(r->r.sob_reg, sob_conv_min), map(r->r.error[], sob_conv_min), sob_plot_args)
        lines!(ax, l2_reg_range, ref_err_repeat; label="Reference", linewidth=3, linestyle=:dash)
        axislegend(position=:lt)
    end
    save(joinpath(@__DIR__, "figs", "reg_conv.png"), fig)
    fig
end

function CreateJointRegTuningPlot(reg_bench_tune)
    fig = Figure()
    N = reg_bench_tune[1].N
    ax = Axis(fig[1,1], xscale=log10, yscale=log10, title=L"Regularization tuning, $N=%$N$", xlabel=L"\lambda_{W^{1,2}}", ylabel=L"\lambda_{L^2}")
    tune_errs = map(r->r.error[], reg_bench_tune)
    sob_vals = map(r->r.sob_reg, reg_bench_tune[1,:])
    l2_vals = map(r->r.l2_reg, reg_bench_tune[:,1])
    hm = heatmap!(ax, sob_vals, l2_vals, log10.(tune_errs))
    Colorbar(fig[1,2], hm, label="log K-S Statistic")
    save(joinpath(@__DIR__, "figs", "reg_tune.png"), fig)
    fig
end
##
qrules = jldopen(joinpath(@__DIR__, "data", "quadrule_error.jld2"))["quadrules"]
CreatePureQuadConvergencePlot(qrules)
CreateHybridQuadConvergencePlot(qrules, :hybrid_mc)
CreateHybridQuadConvergencePlot(qrules, :hybrid_qmc)

##
reg_bench_conv = jldopen(joinpath(@__DIR__, "data", "reg_conv_all.jld2"))["reg_bench_conv"]
reg_bench_tune = jldopen(joinpath(@__DIR__, "data", "reg_tune_all.jld2"))["reg_bench_tune"]
CreateJointRegTuningPlot(reg_bench_tune)
CreateMarginalRegConvergencePlot(reg_bench_conv)