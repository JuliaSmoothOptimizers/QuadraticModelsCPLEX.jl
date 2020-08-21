# QPModelCPLEX

A package to optimize linear and quadratic problems in QPSData format (see
QPSReader.jl)

# Usage

```julia
using QPSReader
qpmodel = readqps("AFIRO.SIF")
stats = optimizeCPlex(qpmodel)
```
