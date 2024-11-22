# QuadraticModelsCPLEX.jl Tutorial

If you dispose from a QuadraticModel `QM` (see the [`Introduction to QuadraticModels.jl`](https://jso-docs.github.io/introduction-to-quadraticmodels/) to learn how to create a QuadraticModel and more), you can solve it with:

```julia
using QuadraticModelsCPLEX
stats = cplex(QM)
```

You can also save the presolved QuadraticModel to a MPS file:

```julia
presolve_to_file(QM; path = "QMpresolved.mps")
```

Then, it is possible to use [`QPSReader.jl`](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) and [`QuadraticModels.jl`](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl) to read the presolved MPS problem:

```julia
qps = readqps("QMpresolved.mps")
QM_ps = QuadraticModel(qps)
```
