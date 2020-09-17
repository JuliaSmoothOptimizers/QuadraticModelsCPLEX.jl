using QPSReader, QuadraticModels, QuadraticModelsCPLEX
using Test

@testset "QuadraticModelsCPLEX.jl" begin
    qps1 = readqps("QAFIRO.SIF") #lower bounds
    stats1 = cplex(QuadraticModel(qps1))
    @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
    @test stats1.status == :acceptable

    qps2 = readqps("HS21.SIF") # low/upp bounds
    stats2 = cplex(QuadraticModel(qps2))
    @test isapprox(stats2.objective, -9.99599999e1, atol=1e-2)
    @test stats2.status == :acceptable

    qps3 = readqps("HS52.SIF") # free bounds
    stats3 = cplex(QuadraticModel(qps3))
    @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
    @test stats3.status == :acceptable
end
