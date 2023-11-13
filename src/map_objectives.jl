abstract type QuadRule end

# Quadrature needs to be more carful in multiple dimensions
struct MCQuad{T} <: QuadRule
    samples::Vector{T}
    function MCQuad(samples::Vector{U}) where {U}
        new{U}(samples)
    end
end

struct BlackboxQuad{T} <: QuadRule
    eval::T
    num_quad::Int
end

GetQuad(q::QuadRule) = __notImplement(GetQuad, typeof(q), QuadRule)
GetQuad(q::MCQuad) = (q.samples, ones(length(q.samples)))
GetQuad(q::BlackboxQuad) = q.eval(q.num_quad)

abstract type LossFunction end
struct KLDiv{T,U} <: LossFunction
    logdensity::T
    gradlogdensity::U
    function KLDiv(logdensity::T1, gradlogdensity::U1 = nothing) where {T1,U1}
        new{T1,U1}(logdensity,gradlogdensity)
    end
end
struct ParamL2Reg <: LossFunction end
# Imitates W_{1,2}, i.e. ||T||_{W_{1,2}}^2 = ||T||_{L_2}^2 + ||T'||_{L_2}^2
struct Sobolev12Reg <: LossFunction end

struct CombinedLoss{P,PReg,SReg} <: LossFunction where {P<:LossFunction, PReg<:LossFunction, SReg<:LossFunction}
    weight_primary::Float64
    eval_primary::P
    weight_p_reg::Float64
    eval_p_reg::PReg
    weight_sob_reg::Float64
    eval_sob_reg::SReg
    function CombinedLoss(primary::P1, param_reg::PReg1 = ParamL2Reg(), sob_reg::SReg1 = Sobolev12Reg(); weight_primary = 1.0, weight_param_reg = 0., weight_sobolev_reg = 0.) where {P1,PReg1,SReg1}
        new{P1,PReg1,SReg1}(weight_primary, primary, weight_param_reg, param_reg, weight_sobolev_reg, sob_reg)
    end
end

Loss(loss::LossFunction, ::TransportMap, ::QuadRule) = __notImplement(Loss, typeof(loss), LossFunction)
LossParamGrad(loss::LossFunction, ::TransportMap, ::QuadRule) = __notImplement(LossParamGrad, typeof(loss), LossFunction)

function Loss(kl::KLDiv, Umap::TransportMap, qrule::QuadRule)
    pts, wts = GetQuad(qrule)
    u_eval = EvaluateMap(Umap, pts)
    logq_comp_u = kl.logdensity(u_eval)
    logdet_u = LogDeterminant(Umap, pts)
    -sum((logq_j + logdet_j)*w_j for (logq_j, logdet_j, w_j) in zip(logq_comp_u, logdet_u, wts))
end

function LossParamGrad(kl::KLDiv, Umap::TransportMap, qrule::QuadRule)
    pts, wts = GetQuad(qrule)
    u_eval, u_pgrad = EvaluateMap(Umap, pts, [DerivativeFlags.None, DerivativeFlags.ParamGrad])
    logq_comp_u_pgrad = kl.gradlogdensity(u_eval)
    logdet_u_pgrad = LogDeterminantParamGrad(Umap, pts)
    ret = [-sum((u_pgrad[i,j]*logq_comp_u_pgrad[j] + logdet_u_pgrad[i,j])*wts[j] for j in axes(u_pgrad,2)) for i in axes(u_pgrad,1)]
    ret
end

function Loss(::ParamL2Reg, Umap::TransportMap, ::QuadRule)
    params = GetParams(Umap)
    sum(p*p for p in params)
end

function LossParamGrad(::ParamL2Reg, Umap::TransportMap, ::QuadRule)
    params = GetParams(Umap)
    2*params
end

function Loss(::Sobolev12Reg, Umap::TransportMap, qrule::QuadRule)
    pts, wts = GetQuad(qrule)
    u_eval, u_igrad = EvaluateMap(Umap, pts, [DerivativeFlags.None, DerivativeFlags.InputGrad])
    ret = 0
    for j in eachindex(u_eval)
        ret += (u_eval[j]*u_eval[j] + u_igrad[j]*u_igrad[j])*wts[j]
    end
    ret
end

function LossParamGrad(::Sobolev12Reg, Umap::TransportMap, qrule::QuadRule)
    pts, wts = GetQuad(qrule)
    u_eval, u_igrad, u_pgrad, u_mgrad = EvaluateMap(Umap, pts, [DerivativeFlags.None, DerivativeFlags.InputGrad, DerivativeFlags.ParamGrad, DerivativeFlags.MixedGrad])
    ret = zeros(NumParams(Umap))
    for i in eachindex(ret)
        for j in eachindex(wts)
            ret[i] += 2*(u_eval[j]*u_pgrad[i,j] + u_igrad[j]*u_mgrad[i,j])*wts[j]
        end
    end
    ret
end

function Loss(loss::CombinedLoss, Umap::TransportMap, qrule::QuadRule)
    primary = Loss(loss.eval_primary, Umap, qrule)
    param_reg = Loss(loss.eval_p_reg, Umap, qrule)
    sob_reg = Loss(loss.eval_sob_reg, Umap, qrule)
    primary*loss.weight_primary + param_reg*loss.weight_p_reg + sob_reg*loss.weight_sob_reg
end

function LossParamGrad(loss::CombinedLoss, Umap::TransportMap, qrule::QuadRule)
    primary = LossParamGrad(loss.eval_primary, Umap, qrule)
    param_reg = LossParamGrad(loss.eval_p_reg, Umap, qrule)
    sob_reg = LossParamGrad(loss.eval_sob_reg, Umap, qrule)
    primary*loss.weight_primary + param_reg*loss.weight_p_reg + sob_reg*loss.weight_sob_reg
end