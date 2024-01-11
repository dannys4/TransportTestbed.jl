abstract type ParentT end
struct ChildT <: ParentT end
foo(x::ParentT) = throw(NotImplementedError(ParentT, foo, typeof(x)))

function TestUtils()
    @test NotImplementedError(ParentT, foo, ChildT) isa Exception
    if VERSION >= v"1.8.0"
        @test_throws "Class ChildT does not implement method foo from parent ParentT" foo(
            ChildT()
        )
    else
        @test_throws NotImplementedError foo(ChildT())
    end
end
