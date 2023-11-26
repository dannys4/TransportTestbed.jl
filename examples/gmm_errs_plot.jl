include("gmm_base.jl")

using DataFrames, JLD2, GLMakie

COLORS = Makie.wong_colors()

# prop_mc or prop_qmc => args
HYBRID_PLOT_DICT = Dict(
    0.10 => QuadRulePlotArgs("10%", COLORS[1], :rect),
    0.25 => QuadRulePlotArgs("25%", COLORS[2], :circle),
    0.50 => QuadRulePlotArgs("50%", COLORS[3], :utriangle),
    0.75 => QuadRulePlotArgs("75%", COLORS[4], :star4),
    0.90 => QuadRulePlotArgs("90%", COLORS[5], :diamond),
)

PLOT_DICT = Dict(
    :quadrature => QuadRulePlotArgs("Gauss-Hermite", COLORS[1], :rect),
    :montecarlo => QuadRulePlotArgs("Monte Carlo", COLORS[2], :circle),
    :qmc => QuadRulePlotArgs("Mapped Sobol QMC", COLORS[3], :utriangle),
    :hybrid_mc => nothing,
    :hybrid_qmc => nothing,
)

module QruleIndex
@enum __QruleIndex Quadrature = 1 MonteCarlo = 2 QMC = 3 HybridMCStart = 4 HybridMCEnd = 8 HybridQMCStart =
    9 HybridQMCEnd = 13
end

function CreatePureConvergencePlot(qrules)
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
        plot_args = PLOT_DICT[conv[1].name]
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

function CreateHybridConvergencePlot(qrules, which_mc)
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
        plot_args = HYBRID_PLOT_DICT[prop_mc]
        Ns = [c.N for c in col]
        errs = [c.error[] for c in col]
        plot_lamb(Ns, errs, plot_args)
    end
    col = qrules[:,Int(QruleIndex.Quadrature)]
    plot_args = PLOT_DICT[:quadrature]
    Ns = [c.N for c in col]
    errs = [c.error[] for c in col]
    plot_lamb(Ns, errs, plot_args, linestyle=:dash)
    axislegend()
    save(joinpath(@__DIR__, "figs", "hybric_conv_$(title_suff).png"), fig)
    fig
end

##
qrules = jldopen(joinpath(@__DIR__, "data", "quadrule_error.jld2"))["quadrules"]
CreatePureConvergencePlot(qrules)
CreateHybridConvergencePlot(qrules, :hybrid_mc)
CreateHybridConvergencePlot(qrules, :hybrid_qmc)