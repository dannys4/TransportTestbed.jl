function SetParams(linmap::LinearMap, params::__VR)::Nothing
    linmap.__coeff .= params
    nothing
end

GetParams(linmap::LinearMap)::Vector{Float64} = linmap.__coeff

NumParams(linmap::LinearMap)::Int = length(linmap.__coeff)

function Evaluate(f::LinearMap, points::__VR)
    evals = EvaluateAll(f.__map_eval, length(f.__coeff) - 1, points)
    evals'f.__coeff
end

function EvaluateInputGrad(f::LinearMap, points::__VR)
    evals, deriv = Derivative(f.__map_eval, length(f.__coeff) - 1, points)
    evals'f.__coeff, deriv'f.__coeff
end

function EvaluateParamGrad(f::LinearMap, points::__VR)
    evals = EvaluateAll(f.__map_eval, length(f.__coeff) - 1, points)
    evals'f.__coeff, evals
end

function EvaluateInputParamGrad(f::LinearMap, points::__VR)
    evals, deriv = Derivative(f.__map_eval, length(f.__coeff) - 1, points)
    evals'f.__coeff, deriv'f.__coeff, evals
end

function EvaluateInputGradHess(f::LinearMap, points::__VR)
    evals, diff, diff2 = SecondDerivative(f.__map_eval, length(f.__coeff) - 1, points)
    evals'f.__coeff, diff'f.__coeff, diff2'f.__coeff
end

function EvaluateInputGradMixedGrad(f::LinearMap, points::__VR)
    evals, diff, = Derivative(f.__map_eval, length(f.__coeff) - 1, points)
    evals'f.__coeff, diff'f.__coeff, diff
end

function EvaluateParamGradInputGradMixedGradMixedInputHess(f::LinearMap, points::__VR)
    evals, diff, diff2 = SecondDerivative(f.__map_eval, length(f.__coeff) - 1, points)
    evals'f.__coeff, evals, diff'f.__coeff, diff, diff2, diff2'f.__coeff
end

function EvaluateMap(f::LinearMap, points::__VR, deriv_flags::__VF)
    length(deriv_flags) == 0 && return []
    max_deriv = InputDerivatives(deriv_flags)
    eval = diff = diff2 = nothing
    if max_deriv == 0
        eval = EvaluateAll(f.__map_eval, length(f.__coeff) - 1, points)
    elseif max_deriv == 1
        eval, diff = Derivative(f.__map_eval, length(f.__coeff) - 1, points)
    elseif max_deriv == 2
        eval, diff, diff2 = SecondDerivative(f.__map_eval, length(f.__coeff) - 1, points)
    else
        throw(ArgumentError("Invalid number of derivatives, $max_deriv"))
    end
    ret = VecOrMat{Float64}[]
    for flag in deriv_flags
        if flag == DerivativeFlags.None
            @assert !isnothing(eval) ""
            push!(ret, eval'f.__coeff)
        elseif flag == DerivativeFlags.InputGrad
            @assert !isnothing(diff) ""
            push!(ret, diff'f.__coeff)
        elseif flag == DerivativeFlags.InputHess
            @assert !isnothing(diff2) ""
            push!(ret, diff2'f.__coeff)
        elseif flag == DerivativeFlags.ParamGrad
            @assert !isnothing(eval) ""
            push!(ret, eval)
        elseif flag == DerivativeFlags.MixedGrad
            @assert !isnothing(diff) ""
            push!(ret, diff)
        elseif flag == DerivativeFlags.MixedHess
            @assert !isnothing(diff2) ""
            push!(ret, diff2)
        else
            throw(ArgumentError("Could not find flag $flag"))
        end
    end
    ret
end
