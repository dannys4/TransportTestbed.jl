using TransportTestbed
using Test, Random
include("test_map_creation.jl")
include("test_map_eval.jl")
include("test_linear_map.jl")
@testset "TransportTestbed.jl" begin
    @testset "Sigmoid Map" begin
        @testset "Map Creation" begin
            TestMapCreation()
        end
        @testset "Map Evaluation" begin
            TestMapEvaluation()
        end
        @testset "Map Derivative" begin
            TestMapDerivative()
        end
        @testset "Map Second Derivative" begin
            TestMapSecondDerivative()
        end
    end
    @testset "Linear Map" begin
        @testset "Identity Map" begin
            TestIdentityMap()
            TestLinearIdentityMap()
        end
    end
end
