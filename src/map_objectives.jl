GetQuad(q::QuadRule) = __notImplement(GetQuad, typeof(q), QuadRule)
GetQuad(q::MCQuad) = (q.samples, fill(1/length(q.samples),length(q.samples)))
GetQuad(q::BlackboxQuad) = q.eval(q.num_quad)
function GetQuad(q::QuadPair)
    pts1, wts1 = GetQuad(q.quad1)
    pts2, wts2 = GetQuad(q.quad2)
    vcat(pts1, pts2), vcat(q.quad1_weight*wts1,q.quad2_weight*wts2)
end

function Loss(loss::LossFunction, ::TransportMap, ::QuadRule)
    __notImplement(Loss, typeof(loss), LossFunction)
end
function LossParamGrad(loss::LossFunction, ::TransportMap, ::QuadRule)
    __notImplement(LossParamGrad, typeof(loss), LossFunction)
end

function Loss(kl::KLDiv, Umap::TransportMap, qrule::QuadRule)
    pts, wts = GetQuad(qrule)
    u_eval = EvaluateMap(Umap, pts)
    logq_comp_u = kl.logdensity(u_eval)
    logdet_u = LogDeterminant(Umap, pts)
    -sum((logq_j + logdet_j) * w_j for
         (logq_j, logdet_j, w_j) in zip(logq_comp_u, logdet_u, wts))
end

function LossParamGrad(kl::KLDiv, Umap::TransportMap, qrule::QuadRule)
    pts, wts = GetQuad(qrule)
    u_eval, u_pgrad = EvaluateMap(Umap,
        pts,
        [DerivativeFlags.None, DerivativeFlags.ParamGrad])
    logq_comp_u_pgrad = kl.gradlogdensity(u_eval)
    logdet_u_pgrad = LogDeterminantParamGrad(Umap, pts)
    ret = [-sum((u_pgrad[i, j] * logq_comp_u_pgrad[j] + logdet_u_pgrad[i, j]) * wts[j] for
                j in axes(u_pgrad, 2)) for i in axes(u_pgrad, 1)]
    ret
end

function Loss(::ParamL2Reg, Umap::TransportMap, ::QuadRule)
    params = GetParams(Umap)
    sum(p * p for p in params)
end

function LossParamGrad(::ParamL2Reg, Umap::TransportMap, ::QuadRule)
    params = GetParams(Umap)
    2 * params
end

function Loss(::Sobolev12Reg, Umap::TransportMap, qrule::QuadRule)
    pts, wts = GetQuad(qrule)
    u_eval, u_igrad = EvaluateMap(Umap,
        pts,
        [DerivativeFlags.None, DerivativeFlags.InputGrad])
    ret = 0
    for j in eachindex(u_eval)
        ret += (u_eval[j] * u_eval[j] + u_igrad[j] * u_igrad[j]) * wts[j]
    end
    ret
end

function LossParamGrad(::Sobolev12Reg, Umap::TransportMap, qrule::QuadRule)
    pts, wts = GetQuad(qrule)
    u_eval, u_igrad, u_pgrad, u_mgrad = EvaluateMap(Umap,
        pts,
        [
            DerivativeFlags.None,
            DerivativeFlags.InputGrad,
            DerivativeFlags.ParamGrad,
            DerivativeFlags.MixedGrad,
        ])
    ret = zeros(NumParams(Umap))
    for i in eachindex(ret)
        for j in eachindex(wts)
            ret[i] += 2 * (u_eval[j] * u_pgrad[i, j] + u_igrad[j] * u_mgrad[i, j]) * wts[j]
        end
    end
    ret
end

function Loss(loss::CombinedLoss, Umap::TransportMap, qrule::QuadRule)
    primary = Loss(loss.eval_primary, Umap, qrule)
    param_reg = Loss(loss.eval_p_reg, Umap, qrule)
    sob_reg = Loss(loss.eval_sob_reg, Umap, qrule)
    primary * loss.weight_primary +
    param_reg * loss.weight_p_reg +
    sob_reg * loss.weight_sob_reg
end

function LossParamGrad(loss::CombinedLoss, Umap::TransportMap, qrule::QuadRule)
    primary = LossParamGrad(loss.eval_primary, Umap, qrule)
    param_reg = LossParamGrad(loss.eval_p_reg, Umap, qrule)
    sob_reg = LossParamGrad(loss.eval_sob_reg, Umap, qrule)
    primary * loss.weight_primary +
    param_reg * loss.weight_p_reg +
    sob_reg * loss.weight_sob_reg
end
