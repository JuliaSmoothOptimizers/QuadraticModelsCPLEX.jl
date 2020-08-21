module QPModelCPLEX

using CPLEX
using QuadraticModels
using NLPModels
using SolverTools
using LinearAlgebra
using SparseArrays

function createQuadraticModel(qpdata)
    return QuadraticModel(qpdata.c, qpdata.qrows, qpdata.qcols, qpdata.qvals,
            Arows=qpdata.arows, Acols=qpdata.acols, Avals=qpdata.avals,
            lcon=qpdata.lcon, ucon=qpdata.ucon, lvar=qpdata.lvar, uvar=qpdata.uvar,
            c0=qpdata.c0)
end


function optimizeCPLEX(QM; method=4, presolve=true, scaling=true, crossover=true,
                       time_limit=3600, output=1)
    SM = SlackModel(QM)

    env = CPLEX.Env()

    CPLEX.set_param!(env, "CPXPARAM_ScreenOutput", output)  # Enable output (0=off)
    CPLEX.set_param!(env, "CPXPARAM_TimeLimit", 3600)  # Time limit
    CPLEX.set_param!(env, "CPXPARAM_Threads", 1)  # Single thread
    if !presolve
        CPLEX.set_param!(env, "CPXPARAM_Preprocessing_Presolve", 0)  # No presolve
    end
    if !scaling
        CPLEX.set_param!(env, "CPXPARAM_Read_Scale", -1) # no scaling
    end
    if !crossover
        CPLEX.set_param!(env, "CPXPARAM_SolutionType", 2)  # No crossover
    end
    CPLEX.set_param!(env, "CPXPARAM_LPMethod", method)  # 4 = Use barrier
    CPLEX.set_param!(env, "CPXPARAM_QPMethod", method) # 4 = Use barrier, 0 = automatic


    Aeq = jac(SM, SM.meta.x0)
    beq = SM.meta.lcon
    H = hess(SM, zeros(length(SM.meta.x0)))
    H = sparse(Symmetric(H, :L))
    f = grad(SM, zeros(length(SM.meta.x0)))
    n,m = size(Aeq)

    model = CPLEX.cplex_model(env; H=H, f = f,
                Aeq = Aeq, beq = beq,
                lb = SM.meta.lvar, ub = SM.meta.uvar)

    t = @timed begin
        CPLEX.optimize!(model)
    end

    x = CPLEX.get_solution(model)
    y = CPLEX.get_constr_duals(model)
    s = CPLEX.get_reduced_costs(model)

    if model.terminator == [0]
        status = :acceptable
    else
        status = :unknown
    end

    stats = GenericExecutionStats(status, SM, solution = x[1:SM.meta.nvar],
                                  objective = CPLEX.get_objval(model),
                                  primal_feas = norm(Aeq * x - beq, Inf),
                                  dual_feas = norm(Aeq' * y - H * x + s - f, Inf),
                                  iter = Int64(CPLEX.c_api_getbaritcnt(model)),
                                  multipliers = y,
                                  elapsed_time = t[2])

    return stats
end

function optimizeCPLEX(qpdata::QPSData; method=0, presolve=true, scaling=true, crossover=true,
                       time_limit=3600, output=1)
    return optimizeCPLEX(createQuadraticModel(qpdata), method=method, presolve=presolve,
                         crossover=crossover, time_limit=time_limit, output=output)
end

end
