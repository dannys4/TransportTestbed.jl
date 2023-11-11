abstract type ParentT end
struct ChildT <: ParentT end
foo(x::ParentT) = throw(NotImplementedError(ParentT, foo, typeof(x)))

function TestUtils()
    pts, wts = gaussprobhermite(50)
    @test abs(pts'wts) < 1e-9
    @test abs(((pts.^2)'*wts) .- 1) < 1e-9
    @test NotImplementedError(ParentT, foo, ChildT) isa Exception
    if VERSION >= v"1.8.0"
        @test_throws "Class ChildT does not implement method foo from parent ParentT" foo(ChildT())
    else
        @test_throws NotImplementedError foo(ChildT())
    end
end