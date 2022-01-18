module QuadraticModelsCPLEX

using CPLEX
using QuadraticModels, SolverCore, SparseMatricesCOO
using LinearAlgebra, SparseArrays

export cplex, presolve_to_file, CPXPresolveException

const cplex_statuses = Dict(1 => :acceptable,
                            2 => :unbounded,
                            3 => :infeasible,
                            4 => :infeasible,
                            10 => :max_iter,
                            11 => :max_time,
                            12 => :exception,
                            13 => :user)

function sparse_csr(I, J, V, m=maximum(I), n=maximum(J))
  csrrowptr = zeros(Int, m+1)
  # Compute the CSR form's row counts and store them shifted forward by one in csrrowptr
  coolen = length(I)
  min(length(J), length(V)) >= coolen || throw(ArgumentError("J and V need length >= length(I) = $coolen"))
  @inbounds for k in 1:coolen
    Ik = I[k]
    if 1 > Ik || m < Ik
      throw(ArgumentError("row indices I[k] must satisfy 1 <= I[k] <= m"))
    end
    csrrowptr[Ik+1] += 1
  end

  # Compute the CSR form's rowptrs and store them shifted forward by one in csrrowptr
  countsum = 1
  csrrowptr[1] = 1
  @inbounds for i in 2:(m+1)
    overwritten = csrrowptr[i]
    csrrowptr[i] = countsum
    countsum += overwritten
  end

  # Counting-sort the column and nonzero values from J and V into csrcolval and csrnzval
  # Tracking write positions in csrrowptr corrects the row pointers
  csrcolval = zeros(Int, length(I))
  csrnzval = zeros(length(I))
  @inbounds for k in 1:coolen
    Ik, Jk = I[k], J[k]
    if 1 > Jk || n < Jk
        throw(ArgumentError("column indices J[k] must satisfy 1 <= J[k] <= n"))
    end
    csrk = csrrowptr[Ik+1]
    csrrowptr[Ik+1] = csrk + 1
    csrcolval[csrk] = Jk
    csrnzval[csrk] = V[k]
  end
  csrrowptr = csrrowptr[1:end-1]
  return csrrowptr, csrcolval, csrnzval
end

cplex_inputQM(QM::QuadraticModel{T, S}; kwargs...) where {T, S} = cplex_inputQM(
  convert(QuadraticModel{T, S, SparseMatrixCOO{T, Int}, SparseMatrixCOO{T, Int}}, QM)
)

function cplex_inputQM(QM::QuadraticModel{T, S, M1, M2}; method=1, display=1,
                       kwargs...) where {T, S, M1 <: SparseMatrixCOO, M2 <: SparseMatrixCOO}

  env = CPLEX.Env()
  CPXsetintparam(env, CPXPARAM_ScreenOutput, display)   # Enable output (0=off)
  CPXsetdblparam(env, CPXPARAM_TimeLimit, 3600)  # Time limit
  CPXsetintparam(env, CPXPARAM_Threads, 1) # Single thread
  for (k, v) in kwargs
    if k==:presolve
      CPXsetintparam(env, CPXPARAM_Preprocessing_Presolve, v) # 0 = no presolve
    elseif k==:scaling
      CPXsetintparam(env, CPXPARAM_Read_Scale, -1) # -1 = no scaling
    elseif k==:crossover
      CPXsetintparam(env, CPXPARAM_SolutionType, v)  # 2 = no crossover
    end
  end
  CPXsetintparam(env, CPXPARAM_LPMethod, method)  # 4 = Use barrier
  CPXsetintparam(env, CPXPARAM_QPMethod, method) # 4 = Use barrier, 0 = automatic

  status_p = Ref{Cint}()
  lp = CPXcreateprob(env, status_p, "")
  CPXnewcols(env, lp, QM.meta.nvar, QM.data.c, QM.meta.lvar, QM.meta.uvar, C_NULL, C_NULL)
  CPXchgobjoffset(env, lp, QM.data.c0)
  if QM.meta.nnzh > 0
    Q = sparse(QM.data.H.rows, QM.data.H.cols, QM.data.H.vals, QM.meta.nvar, QM.meta.nvar)
    diag_matrix = spdiagm(0 => diag(Q))
    Q = Q + Q' - diag_matrix
    qmatcnt = zeros(Int, QM.meta.nvar)
    for k = 1:QM.meta.nvar
      qmatcnt[k] = Q.colptr[k+1] - Q.colptr[k]
    end
    CPXcopyquad(env, lp, convert(Array{Cint,1}, Q.colptr[1:end-1].-1), convert(Array{Cint,1}, qmatcnt),
                convert(Array{Cint,1}, Q.rowval.-1), Q.nzval)
  end

  Acsrrowptr, Acsrcolval, Acsrnzval = sparse_csr(QM.data.A.rows, QM.data.A.cols,
                                                  QM.data.A.vals, QM.meta.ncon,
                                                  QM.meta.nvar)

  sense = fill(Cchar('A'), QM.meta.ncon) # lower, greater, range or equal. A is for the init
  if length(QM.meta.jinf) > 0
    error("infeasible bounds in A")
  end
  p_low, p_upp, p_rng, p_fix = 1, 1, 1, 1
  for j=1:QM.meta.ncon
    if length(QM.meta.jlow) > 0 && QM.meta.jlow[p_low] == j
      sense[j] = Cchar('G')
      if (p_low < length(QM.meta.jlow)) p_low += 1 end
    elseif length(QM.meta.jupp) > 0 && QM.meta.jupp[p_upp] == j
      sense[j] = Cchar('L')
      if (p_upp < length(QM.meta.jupp)) p_upp += 1 end
    elseif length(QM.meta.jrng) > 0 && QM.meta.jrng[p_rng] == j
      sense[j] = Cchar('R')
      if (p_rng < length(QM.meta.jrng)) p_rng += 1 end
    elseif length(QM.meta.jfix) > 0 && QM.meta.jfix[p_fix] == j
      sense[j] = Cchar('E')
      if (p_fix < length(QM.meta.jfix)) p_fix += 1 end
    else
      error("A error")
    end
  end
  rhs = zeros(QM.meta.ncon)
  drange = zeros(QM.meta.ncon)
  for j = 1:QM.meta.ncon
    if QM.meta.lcon[j] != -Inf && QM.meta.ucon[j] != Inf
      rhs[j] = QM.meta.ucon[j]
      drange[j] = QM.meta.ucon[j] - QM.meta.lcon[j]
    elseif QM.meta.lcon[j] != -Inf && QM.meta.ucon[j] == Inf
      rhs[j] = QM.meta.lcon[j]
    elseif QM.meta.lcon[j] == -Inf && QM.meta.ucon[j] != Inf
      rhs[j] = QM.meta.ucon[j]
    else
      rhs[j] = Inf
    end
  end
  CPXaddrows(env, lp, 0, QM.meta.ncon, length(Acsrcolval), rhs,
             sense, convert(Vector{Cint}, Acsrrowptr.- Cint(1)), convert(Vector{Cint}, Acsrcolval.- Cint(1)),
             Acsrnzval, C_NULL, C_NULL)

  return env, lp
end

function cplex(QM::QuadraticModel; method=4, display=1, kwargs...)

  env, lp = cplex_inputQM(QM; method=method, display=display, kwargs...)

  t = @timed begin
    if QM.meta.nnzh > 0
      CPXqpopt(env, lp)
    else
      CPXlpopt(env, lp)
    end
  end

  x = Vector{Cdouble}(undef, QM.meta.nvar)
  CPXgetx(env, lp, x, 0, QM.meta.nvar-1)
  y = Vector{Cdouble}(undef, QM.meta.ncon)
  CPXgetpi(env, lp, y, 0, QM.meta.ncon-1)
  s = Vector{Cdouble}(undef, QM.meta.nvar)
  CPXgetdj(env, lp, s, 0, QM.meta.nvar-1)
  primal_feas = Vector{Cdouble}(undef, 1)
  CPXgetdblquality(env, lp, primal_feas, CPX_MAX_PRIMAL_RESIDUAL)
  dual_feas = Vector{Cdouble}(undef, 1)
  CPXgetdblquality(env, lp, dual_feas, CPX_MAX_DUAL_RESIDUAL)
  objval_p = Vector{Cdouble}(undef, 1)
  CPXgetobjval(env, lp, objval_p)

  stats = GenericExecutionStats(get(cplex_statuses, CPXgetstat(env, lp), :unknown),
                                QM, solution = x,
                                objective = objval_p[1],
                                primal_feas = primal_feas[1],
                                dual_feas = dual_feas[1],
                                iter = Int64(CPXgetbaritcnt(env, lp)),
                                multipliers = y,
                                elapsed_time = t[2])
  return stats
end

mutable struct CPXPresolveException <: Exception
  msg::AbstractString
end

function presolve_to_file(QM::QuadraticModel; path::String = "presolved.mps", method=4, display=1, kwargs...)

  env, lp = cplex_inputQM(QM; method=method, display=display, kwargs...)
  objoff = Ref{Cdouble}()
  str_p =  string(path[1:end-3], "pre")
  status = CPXpreslvwrite(env, lp, str_p, objoff)
  status == 1103 && throw(CPXPresolveException("No presolve operation performed"))

  status_p = Ref{Cint}()
  lp_p = CPXcreateprob(env, status_p, "")

  CPXreadcopyprob(env, lp_p, str_p, C_NULL)
  status = CPXwriteprob(env, lp_p, path, C_NULL)
end

end