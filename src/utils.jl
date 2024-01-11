import Base: showerror, Exception
struct NotImplementedError{T} <: Base.Exception where {T}
    parent_class::Type{T}
    method::Function
    child_class::Type{<:T}
    function NotImplementedError(
        parent_class::Type{T1}, method, child_class::Type{<:T1}
    ) where {T1}
        new{T1}(parent_class, method, child_class)
    end
end
function Base.showerror(io::IO, exc::NotImplementedError{T}) where {T}
    print(
        io,
        "Class $(exc.child_class) does not implement method $(exc.method) from parent $(exc.parent_class)",
    )
end

function __notImplement(method, child, parent)
    throw(NotImplementedError(parent, method, child))
end

# A function that implements the Box-Muller algorithm
function BoxMuller(unif_samples::AbstractVector)
    U1 = @view unif_samples[1:(end÷2)]
    U2 = @view unif_samples[(end+3)÷2 : end]
    mult_term = sqrt.(-2*log.(U1))
    z1 = mult_term .* cos.(2*π*U2)
    z2 = mult_term .* sin.(2*π*U2)
    vcat(z1, z2)
end