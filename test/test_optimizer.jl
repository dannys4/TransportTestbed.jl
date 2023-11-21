function TestFakeOptimizer()
    opt = FakeOptimizer()
    l = FakeLossFunction()
    q = FakeQuadRule()
    @test_throws NotImplementedError Optimize(opt, l, q)
end
