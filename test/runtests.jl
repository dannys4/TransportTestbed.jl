using TransportTestbed
using Test
include("test_map_creation.jl")
include("test_map_eval.jl")
@testset "TransportTestbed.jl" begin
    @testset "Map Creation" TestMapCreation()
    @testset "Map Evaluation" TestMapEvaluation()
    @testset "Map Derivative" TestMapDerivative()
    @testset "Map Second Derivative" TestMapSecondDerivative()
end
