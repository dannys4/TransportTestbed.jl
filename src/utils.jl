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
