# Calculates the maximum input derivative based on flags
function InputDerivatives(flags::__VF)
    m_flags = UInt8(maximum(flags))
    m_flags < 1 && return -1
    m_flags < 3 && return 0
    m_flags < 5 && return 1
    return 2
end

EvaluateMap(p::TransportMap, ::__VR, ::__VF) = __notImplement(EvaluateMap, typeof(p), TransportMap)
EvaluateMap(f::TransportMap, points::__VR, deriv_flag::DerivativeFlags.__DerivativeFlags = DerivativeFlags.None) = EvaluateMap(f, points, [deriv_flag])[]

function LogDeterminant(umap::TransportMap, pts::Vector{Float64})
    igrad = EvaluateMap(umap, pts, DerivativeFlags.InputGrad)
    log.(igrad)
end

function LogDeterminantInputGrad(umap::TransportMap, pts::Vector{Float64})
    igrad, ihess = EvaluateMap(umap, pts, [DerivativeFlags.InputGrad, DerivativeFlags.InputHess])
    ihess ./ igrad
end

function LogDeterminantParamGrad(umap::TransportMap, pts::Vector{Float64})
    igrad, mgrad = EvaluateMap(umap, pts, [DerivativeFlags.InputGrad, DerivativeFlags.MixedGrad])
    mgrad ./ igrad'
end