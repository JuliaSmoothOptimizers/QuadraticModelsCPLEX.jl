# QPModelCPLEX

A package to optimize linear and quadratic problems in QPSData format (see
QPSReader.jl)

# Usage

```julia
using QPSReader
using QPModelCPLEX
qpmodel = readqps("AFIRO.SIF")
stats = QPModelCPLEX.optimizeCPLEX(qpmodel)
```
