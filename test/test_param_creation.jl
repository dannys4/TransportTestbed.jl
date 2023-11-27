import TransportTestbed:
    NotImplementedError, Evaluate, EvaluateAll, Derivative, SecondDerivative

function TestFakeParamCreation()
    p = FakeParam()
    s = FakeSigmoid()
    t = FakeTail()
    @test_throws NotImplementedError EvaluateAll(p, 3, [1, 2, 3])
    @test_throws NotImplementedError Derivative(p, 3, [1, 2, 3])
    @test_throws NotImplementedError SecondDerivative(p, 3, [1, 2, 3])
    @test_throws NotImplementedError Evaluate(s, 3.0)
    @test_throws NotImplementedError Derivative(s, 3.0)
    @test_throws NotImplementedError SecondDerivative(s, 3.0)
    @test_throws NotImplementedError Evaluate(t, 3.0)
    @test_throws NotImplementedError Derivative(t, 3.0)
    @test_throws NotImplementedError SecondDerivative(t, 3.0)
end

function TestSigmoidParamCreation()
    centers = [[1], [2, 3]]
    widths = [[1]]
    weights = [[1]]
    e = nothing
    try
        SigmoidParam{Logistic,SoftPlus}(centers, widths, weights, -1, 1, 1, 1)
    catch ee
        if ee isa ArgumentError
            e = ee
        else
            rethrow(ee)
        end
    end
    @test !isnothing(e)
    centers = [[1], [1, 2, 3]]
    widths = [[1], [1, 2, 3]]
    weights = [[1], [1, 2, 3]]
    e = nothing
    try
        SigmoidParam{Logistic,SoftPlus}(centers, widths, weights, -1, 1, 1, 1)
    catch ee
        if ee isa ArgumentError
            e = ee
        else
            rethrow(ee)
        end
    end
    @test !isnothing(e)
    centers = [[0], [-1, 1]]
    widths = [[1], [1, 1]]
    weights = [[1], [0.5, 0.5]]
    e = nothing
    try
        SigmoidParam{Logistic,SoftPlus}(centers, widths, weights, 1, -1, 1, 1)
    catch ee
        if ee isa ArgumentError
            e = ee
        else
            rethrow(ee)
        end
    end
    @test !isnothing(e)
    f = SigmoidParam{Logistic,SoftPlus}(centers, widths, weights, -1, 1, 1, 1)
    @test f.max_order == 4
    f = CreateSigmoidParam(; centers, widths, weights)
    @test f.max_order == 4
end
