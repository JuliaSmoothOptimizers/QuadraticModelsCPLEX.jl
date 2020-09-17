# QPModelCPLEX

A package to use CPLEX to optimize linear and quadratic problems in QPSData
format (see QPSReader.jl)

# Usage

```julia
using QPSReader
using QuadraticModelsCPLEX
qpmodel = readqps("AFIRO.SIF")
stats = QPModelCPLEX.optimizeCPLEX(qpmodel)
```
