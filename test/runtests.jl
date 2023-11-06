using TransportTestbed
using Test
include("test_map_creation.jl")
include("test_map_eval.jl")
@testset "TransportTestbed.jl" begin
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
