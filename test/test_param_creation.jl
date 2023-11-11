function TestSigmoidParamCreation()
    centers = [[1],[2,3]]
    widths = [[1]]
    weights = [[1]]
    e = nothing
    try
        SigmoidParam{Logistic,SoftPlus}(centers, widths, weights, -1, 1)
    catch ee
        if ee isa ArgumentError
            e = ee
        else
            rethrow(ee)
        end
    end
    @test !isnothing(e)
    centers = [[1],[1,2,3]]
    widths = [[1],[1,2,3]]
    weights = [[1],[1,2,3]]
    e = nothing
    try
        SigmoidParam{Logistic,SoftPlus}(centers, widths, weights, -1, 1)
    catch ee
        if ee isa ArgumentError
            e = ee
        else
            rethrow(ee)
        end
    end
    @test !isnothing(e)
    centers = [[0], [-1,1]]
    widths = [[1], [1,1]]
    weights = [[1], [0.5,0.5]]
    e = nothing
    try
        SigmoidParam{Logistic,SoftPlus}(centers, widths, weights, 1, -1)
    catch ee
        if ee isa ArgumentError
            e = ee
        else
            rethrow(ee)
        end
    end
    @test !isnothing(e)
    f = SigmoidParam{Logistic,SoftPlus}(centers, widths, weights, -1, 1)
    @test f.max_order == 4
    f = CreateSigmoidParam(;centers, widths, weights)
    @test f.max_order == 4
end