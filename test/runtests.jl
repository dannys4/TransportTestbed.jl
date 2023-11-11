using TransportTestbed
using Test, Random
include("test_param_creation.jl")
include("test_sigmoid_eval.jl")
include("test_linear_map.jl")
@testset "TransportTestbed.jl" begin
    TestFakeParamCreation()
    @testset "Sigmoid Map Param" begin
        @testset "Sigmoid Creation" begin
            TestSigmoidParamCreation()
        end
        @testset "Sigmoid Evaluation" begin
            TestSigmoidParamEvaluation()
        end
        @testset "Sigmoid Derivative" begin
            TestSigmoidParamDerivative()
        end
        @testset "Sigmoid Second Derivative" begin
            TestSigmoidParamSecondDerivative()
        end
    end
    @testset "Linear Map" begin
        rng = Xoshiro(1028302)
        
        @testset "Identity Map" begin
            TestIdentityMapParam()

            id = IdMapParam()
            num_coeffs = 4
            linmap = LinearMap(id, num_coeffs)

            TestLinearMapEvaluate(linmap, rng)
            TestLinearMapGrad(linmap, rng)
            TestLinearMapHess(linmap, rng)
            TestLinearMapEvaluateMap(linmap, rng)
        end

        @testset "Sigmoid Map" begin
            num_sigs = 10
            centers = [[((2i-1)-j)/(j+1) for i in 1:j] for j in 1:num_sigs]
            sigmap = CreateSigmoidParam(;centers)
            linmap = LinearMap(sigmap, sigmap.max_order+1)
            TestLinearMapEvaluate(linmap, rng)
            TestLinearMapGrad(linmap, rng)
            TestLinearMapHess(linmap, rng)
            TestLinearMapEvaluateMap(linmap, rng)
        end
    end
end
