import Base: showerror, Exception
struct NotImplementedError{T} <: Base.Exception where {T}
    parent_class::Type{T}
    method::Function
    child_class::Type{<:T}
end
NotImplementedError(parent_class::Type{T}, method, child_class::Type{<:T}) where {T} = NotImplementedError{T}(parent_class, method, child_class)
Base.showerror(io::IO, exc::NotImplementedError{T}) where {T} = print(io, "Class $(exc.child_class) does not implement method $(exc.method) from parent $(exc.parent_class)")

function gaussprobhermite(n_pts::Int)
    pts, wts = gausshermite(n_pts)
    pts *= sqrt(2)
    wts /= sqrt(pi)
    pts, wts
end

function __notImplement(method, child, parent)
    throw(NotImplementedError(parent, method, child))
end
