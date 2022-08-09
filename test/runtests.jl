using QPSReader, QuadraticModels, QuadraticModelsCPLEX
using LinearAlgebra
using Test

@testset "QuadraticModelsCPLEX.jl" begin
  qps1 = readqps("QAFIRO.SIF") #lower bounds
  stats1 = cplex(QuadraticModel(qps1), threads = 1)
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

  # convert qp
  Q = [
      6.0 2.0 1.0
      2.0 5.0 2.0
      1.0 2.0 4.0
  ]
  c = [-8.0; -3; -3]
  A = [
  1.0 0.0 1.0
  0.0 2.0 1.0
  ]
  b = [0.0; 3]
  l = [0.0; 0; 0]
  u = [Inf; Inf; Inf]
  qp_dense = QuadraticModel(
      c,
      tril(Q),
      A = A,
      lcon = b,
      ucon = b,
      lvar = l,
      uvar = u,
      c0 = 0.0,
      name = "QM_dense",
  )
  stats_dense = cplex(qp_dense)
  @test isapprox(stats_dense.objective, 1.1249999990782493, atol = 1e-2)
  @test stats_dense.status == :acceptable
end

@testset "presolve" begin
  qps1 = readqps("QAFIRO.SIF") #lower bounds
  presolve_to_file(QuadraticModel(qps1), path = "QAFIRO_PS.mps")
  qpsp1 = readqps("QAFIRO_PS.mps")
  stats1 = cplex(QuadraticModel(qpsp1))
  @test isapprox(stats1.objective, -1.59078179, atol=1e-2)
  @test stats1.status == :acceptable

  qps2 = readqps("HS21.SIF") #lower bounds
  @test_throws CPXPresolveException presolve_to_file(QuadraticModel(qps2), path = "HS21_PS.mps")

  qps3 = readqps("HS52.SIF") #lower bounds
  presolve_to_file(QuadraticModel(qps3), path = "HS52_PS.mps")
  qpsp3 = readqps("HS52_PS.mps")
  stats3 = cplex(QuadraticModel(qpsp3))
  @test isapprox(stats3.objective, 5.32664756, atol=1e-2)
  @test stats3.status == :acceptable
end