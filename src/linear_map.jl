function SetCoeffs(linmap::LinearMap, coeffs::__VR)
    linmap.__coeff .= coeffs
end

GetCoeffs(linmap::LinearMap) = linmap.__coeff

NumCoeffs(linmap::LinearMap) = length(linmap.__coeff)

function Evaluate(f::LinearMap, points::__VR)
    evals = EvaluateAll(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff
end

function EvaluateInputGrad(f::LinearMap, points::__VR)
    evals, deriv = Derivative(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, deriv'f.__coeff
end

function EvaluateParamGrad(f::LinearMap, points::__VR)
    evals = EvaluateAll(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, evals
end

function EvaluateInputParamGrad(f::LinearMap, points::__VR)
    evals, deriv = Derivative(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, deriv'f.__coeff, evals
end

function EvaluateInputGradHess(f::LinearMap, points::__VR)
    evals, diff, diff2 = SecondDerivative(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, diff'f.__coeff, diff2'f.__coeff
end

function EvaluateInputGradMixedGrad(f::LinearMap, points::__VR)
    evals, diff, = Derivative(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, diff'f.__coeff, diff
end

function EvaluateParamGradInputGradMixedGradMixedInputHess(f::LinearMap, points::__VR)
    evals, diff, diff2 = SecondDerivative(f.__map_eval, length(f.__coeff)-1, points)
    evals'f.__coeff, evals, diff'f.__coeff, diff, diff2, diff2'f.__coeff
end

# Calculates the maximum input derivative based on flags
function InputDerivatives(flags::__VF)
    m_flags = UInt8(maximum(flags))
    m_flags < 2 && return 0
    m_flags < 4 && return 1
    return 2
end

function EvaluateMap(f::LinearMap, points::__VR, deriv_flags::__VF)
    length(deriv_flags) == 0 && return []
    max_deriv = InputDerivatives(deriv_flags)
    eval = diff = diff2 = nothing
    if max_deriv == 0
        eval = EvaluateAll(f.__map_eval, length(f.__coeff)-1, points)
    elseif max_deriv == 1
        eval, diff = Derivative(f.__map_eval, length(f.__coeff)-1, points)
    elseif max_deriv == 2
        eval, diff, diff2 = SecondDerivative(f.__map_eval, length(f.__coeff)-1, points)
    else
        @error "Invalid number of derivatives, $max_deriv"
    end
    ret = []
    for flag in deriv_flags
        if flag == DerivativeFlags.None
            push!(ret, eval'f.__coeff)
        elseif flag == DerivativeFlags.InputGrad
            push!(ret, diff'f.__coeff)
        elseif flag == DerivativeFlags.InputHess
            push!(ret, diff2'f.__coeff)
        elseif flag == DerivativeFlags.ParamGrad
            push!(ret, eval)
        elseif flag == DerivativeFlags.MixedGrad
            push!(ret, diff)
        elseif flag == DerivativeFlags.MixedHess
            push!(ret, diff2)
        else
            @error "Could not find flag $flag"
        end
    end
    ret
end
EvaluateMap(f::LinearMap, points::__VR, deriv_flag::DerivativeFlags.__DerivativeFlags = DerivativeFlags.None) = EvaluateMap(f, points, [deriv_flag])[]