module QuadraticModelsCPLEX

using CPLEX
using QuadraticModels
using SolverTools
using SparseArrays

export cplex

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

function cplex(QM; method=4, display=1, kwargs...)

    env = CPLEX.Env()
    CPLEX.set_param!(env, "CPXPARAM_ScreenOutput", display)  # Enable output (0=off)
    CPLEX.set_param!(env, "CPXPARAM_TimeLimit", 3600)  # Time limit
    CPLEX.set_param!(env, "CPXPARAM_Threads", 1)  # Single thread
    # use kwargs change to presolve, scaling and crossover mode
    # example: cplex(QM, presolve=0) (see cplex doc for other options)
    for (k, v) in kwargs
        if k==:presolve
            CPLEX.set_param!(env, "CPXPARAM_Preprocessing_Presolve", v) # 0 = no presolve
        elseif k==:scaling
            CPLEX.set_param!(env, "CPXPARAM_Read_Scale", v) # -1 = no scaling
        elseif k==:crossover
            CPLEX.set_param!(env, "CPXPARAM_SolutionType", v)  # 2 = no crossover
        end
    end
    CPLEX.set_param!(env, "CPXPARAM_LPMethod", method)  # 4 = Use barrier
    CPLEX.set_param!(env, "CPXPARAM_QPMethod", method) # 4 = Use barrier, 0 = automatic

    model = CPLEX.Model(env, "")
    CPLEX.set_sense!(model, :Min)
    CPLEX.add_vars!(model, QM.data.c, QM.meta.lvar, QM.meta.uvar)
    CPLEX.c_api_chgobjoffset(model, QM.data.c0)
    if QM.meta.nnzh > 0
        CPLEX.add_qpterms!(model, QM.data.Hrows, QM.data.Hcols, QM.data.Hvals)
    end

    Acsrrowptr, Acsrcolval, Acsrnzval = sparse_csr(QM.data.Arows,QM.data.Acols,
                                                   QM.data.Avals, QM.meta.ncon,
                                                   QM.meta.nvar)

    CPLEX.add_constrs!(model, Acsrrowptr, Acsrcolval, Acsrnzval, '<',
                            QM.meta.ucon)
    CPLEX.add_constrs!(model, Acsrrowptr, Acsrcolval, Acsrnzval, '>',
                            QM.meta.lcon)

    t = @timed begin
        CPLEX.optimize!(model)
    end

    x = CPLEX.get_solution(model)
    y = CPLEX.get_constr_duals(model)
    # s = CPLEX.get_reduced_costs(model)
    primal_feas = Vector{Cdouble}(undef, 1)
    CPLEX.@cpx_ccall(getdblquality, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}, Cint),
                     model.env.ptr, model.lp, primal_feas, convert(Cint,11))
    dual_feas = Vector{Cdouble}(undef, 1)
    CPLEX.@cpx_ccall(getdblquality, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}, Cint),
                     model.env.ptr, model.lp, dual_feas, convert(Cint,15))

    if model.terminator == [0]
        status = :acceptable
    else
        status = :unknown
    end

    stats = GenericExecutionStats(get(cplex_statuses, CPLEX.get_status_code(model), :unknown),
                                  QM, solution = x,
                                  objective = CPLEX.get_objval(model),
                                  primal_feas = primal_feas[1],
                                  dual_feas = dual_feas[1],
                                  iter = Int64(CPLEX.c_api_getbaritcnt(model)),
                                  multipliers = y,
                                  elapsed_time = t[2])

    return stats
end

end
