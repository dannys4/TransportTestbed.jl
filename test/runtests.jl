using TransportTestbed
using Test, Random, LinearAlgebra, FastGaussQuadrature

include("test_utils.jl")
include("test_types.jl")
include("test_param_creation.jl")
include("test_sigmoid_eval.jl")
include("test_linear_map.jl")
include("test_optim.jl")
include("test_transport_map.jl")

rng = Xoshiro(1028302)

@testset "TransportTestbed.jl" begin
    @testset "Utils" begin
        TestUtils()
    end
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
        
        @testset "Identity Map" begin
            TestIdentityMapParam()
            linmap = DefaultIdentityMap()

            TestLinearMapEvaluate(linmap, rng)
            TestLinearMapGrad(linmap, rng)
            TestLinearMapHess(linmap, rng)
            TestLinearMapEvaluateMap(linmap, rng)
        end

        @testset "Sigmoid Map" begin
            linmap = DefaultSigmoidMap()
            TestLinearMapEvaluate(linmap, rng)
            TestLinearMapGrad(linmap, rng)
            TestLinearMapHess(linmap, rng)
            TestLinearMapEvaluateMap(linmap, rng)
        end
    end
    @testset "TransportMap" begin
        TestFakeTransportMap()
        TestLogDeterminant(DefaultIdentityMap(), rng)
        TestLogDeterminant(DefaultSigmoidMap(), rng)
    end
    @testset "Optimization" begin
        TestFakeOptimization()
        TestKLDiv(rng)
        TestRegularizers()
        TestLossParamGrad()
    end
end
