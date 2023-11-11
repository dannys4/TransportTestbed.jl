abstract type QuadRule end

# Quadrature needs to be more carful in multiple dimensions
struct MCQuad{T} <: QuadRule
    samples::Vector{Float64}
end

struct BlackboxQuad{T} <: QuadRule
    eval::T
end

GetQuad(q::QuadRule, ::Int) = __notImplement(getQuad, typeof(q), QuadRule)
GetQuad(q::MCQuad, N::Int) = (q.samples[1:N], fill(1/N, N))
GetQuad(q::BlackboxQuad, N::Int) = q.eval(N)

abstract type LossFunction end
struct KLDiv <: LossFunction end
struct ParamL2Reg <: LossFunction end
struct Sobolev2Reg <: LossFunction end

struct CombinedLoss{P,PReg,SReg} <: LossFunction where {P<:LossFunction, PReg<:LossFunction, SReg<:LossFunction}
    weight_primary::Float64
    eval_primary::P
    weight_p_reg::Float64
    eval_p_reg::PReg
    weight_sob_reg::Float64
    eval_sob_reg::SReg
    function CombinedLoss(primary::P1, param_reg::PReg1 = ParamL2Reg(), sob_reg::SReg1 = Sobolev2Reg(); weight_primary = 1.0, weight_param_reg = 0., weight_sobolev_reg = 0.) where {P1,PReg1,SReg1}
        new{P1,PReg1,SReg1}(weight_primary, primary, weight_param_reg, param_reg, weight_sobolev_reg, sob_reg)
    end
end

Loss(loss::LossFunction, ::TransportMap, _, ::QuadRule, ::Int) = __notImplement(Loss, typeof(loss), LossFunction)

function Loss(::KLDiv, Umap::TransportMap, logdensity::D, qrule::QuadRule, num_quad::Int) where {D}
    pts, wts = GetQuad(qrule, num_quad)
    u_eval = EvaluateMap(Umap, pts)
    logq_comp_u = logdensity(u_eval)
    logdet_u = LogDeterminant(Umap, logq_comp_u)
    -(logq_comp_u + logdet_u)'*wts
end