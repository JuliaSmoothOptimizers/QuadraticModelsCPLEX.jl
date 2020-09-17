# QuadraticModelsCPLEX

A package to use CPLEX to optimize linear and quadratic problems in QPSData
format (see QPSReader.jl)

# Usage

```julia
using QPSReader, QuadraticModels, QuadraticModelsCPLEX
qps = readqps("AFIRO.SIF")
qm = QuadraticModel(qps)
stats = cplex(qm)
```
