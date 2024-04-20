using Revise
using JSOSolvers, SolverBenchmark, SolverTools, Plots, DataFrames
using NLPModels, NLPModelsModifiers
using CSV
using PartiallySeparableNLPModels

n = 5000 # problem size 
include("problems.jl")

const max_time = 20.0
const max_eval = 5000

# warm-up
solver = Dict{Symbol,Function}(
  :PBFGS => ((prob;kwargs...) -> JSOSolvers.trunk(PBFGSNLPModel(prob); monotone=true, kwargs...)),
  :PLBFGS => ((prob;kwargs...) -> JSOSolvers.trunk(PLBFGSNLPModel(prob); monotone=true, kwargs...)),
  :PSR1 => ((prob;kwargs...) -> JSOSolvers.trunk(PSR1NLPModel(prob); monotone=true, kwargs...)),
  :PLSR1 => ((prob;kwargs...) -> JSOSolvers.trunk(PLSR1NLPModel(prob); monotone=true, kwargs...)),
  :PSE => ((prob;kwargs...) -> JSOSolvers.trunk(PSENLPModel(prob); monotone=true, kwargs...)),
  :PLSE => ((prob;kwargs...) -> JSOSolvers.trunk(PLSENLPModel(prob); monotone=true, kwargs...)),
  :PCS => ((prob;kwargs...) -> JSOSolvers.trunk(PCSNLPModel(prob); monotone=true, kwargs...)),
  :PHv => ((prob;kwargs...) -> JSOSolvers.trunk(PSNLPModel(prob); monotone=true, kwargs...)),
  :LBFGS_F => ((prob;kwargs...) -> JSOSolvers.trunk(LBFGSModel(prob); monotone=true, kwargs...)),
  :LBFGS => ((prob;kwargs...) -> JSOSolvers.lbfgs(prob; kwargs...)),
  :LSR1_F => ((prob;kwargs...) -> JSOSolvers.trunk(LSR1Model(prob); monotone=true, kwargs...)),
  :Hv => ((prob;kwargs...) -> JSOSolvers.trunk(prob; monotone=true, kwargs...)),
)

ENV["JULIA_DEBUG"] = SolverBenchmark
stats = bmark_solvers(solver, problems; max_time, max_eval, verbose=0)

keys_stats = keys(stats)

path_result = pwd()*"/script/profiles/results/"

println("print tables")
# select relevant fields
selected_fields = [:name, :nvar, :elapsed_time, :iter, :dual_feas, :status, :objective, :neval_obj, :neval_grad, :neval_hprod, :obj_5grad_5Hv, :time_sur_obj_5grad_5Hv, :inverse_pourcentage_pas_accepte]
for i in keys_stats
  println(stdout, "\n" * string(i) )
  pretty_stats(stdout, stats[i][!, [:name, :nvar, :elapsed_time, :iter, :dual_feas, :status, :objective, :neval_obj, :neval_grad, :neval_hprod]], tf=tf_markdown)
end

println("save markdown table")
location_md = string(path_result*"tables.md")
io = open(location_md,"w")
for i in keys_stats
  println(io, "\n" * string(i) )
  pretty_stats(io, stats[i][!, [:name, :nvar, :elapsed_time, :iter, :dual_feas, :status, :objective, :neval_obj, :neval_grad, :neval_hprod]], tf=tf_markdown)
end
close(io)

solved(df) = first_order(df) .| unbounded(df)
first_order(df) = df.status .== :first_order
unbounded(df) = df.status .== :unbounded
cost_iter(df) = .!solved(df) .* Inf .+ df.iter
cost_time(df) = .!solved(df) .* Inf .+ df.elapsed_time
cost_obj_5grad(df) = .!solved(df) .* Inf .+ df.neval_obj .+ 5 .* df.neval_grad

# save DataFrame
for key in keys_stats
  CSV.write(path_result*"dataframes/"*string(key)*".csv", stats[key])
end 


#= save latex tables =#
println("print latex table")
location_latex = string(path_result*"tables.tex")
io = open(location_latex,"w")
for i in keys_stats
  println(io, "\n" * string(i) )
  pretty_latex_stats(io, stats[i][!, [:name, :nvar, :elapsed_time, :iter, :dual_feas, :status, :objective, :neval_obj, :neval_grad, :neval_hprod]])
end
close(io)

println("print profiles")
ENV["GKSwstype"]=100 # needed for disable print on servers 

p_iter = SolverBenchmark.performance_profile(stats, df -> cost_iter(df); legend=:bottomright )
savefig(p_iter, path_result*"total/iter_profile.pdf")
p_time = SolverBenchmark.performance_profile(stats, df -> cost_time(df); legend=:bottomright )
savefig(p_time, path_result*"total/time_profile.pdf")



# Lecture only
# memorize_keys_stats = [:PLSR1, :PBFGS, :PSE, :PLSE, :PLBFGS, :TRUNK, :PSR1, :PCS, :PS, :LBFGS_F, :TRUNK_LSR1]

# new_stats = Dict{Symbol, DataFrame}()
# for key in memorize_keys_stats
#   new_stats[key] = DataFrame(CSV.File(path_result*"dataframes/"*string(key)*".csv"))
# end 


# Quasi-Newton profiles
QN_Hv_keys = [:Hv, :PHv, :LBFGS_F, :LSR1_F, :LBFGS]
QN_Hv_stats = Dict{Symbol, DataFrame}()
for key in QN_Hv_keys
  QN_Hv_stats[key] = stats[key]
end 
path_result_QN_Hv = path_result*"Qn-Hv/"
p_iter = SolverBenchmark.performance_profile(QN_Hv_stats, df -> cost_iter(df); legend=:bottomright)
savefig(p_iter, path_result_QN_Hv*"iter_profile.pdf")
p_time = SolverBenchmark.performance_profile(QN_Hv_stats, df -> cost_time(df); legend=:bottomright)
savefig(p_time, path_result_QN_Hv*"time_profile.pdf")



# Partitioned quasi-Newton profiles
PQN_keys = [:PBFGS, :PSE, :PSR1, :PCS]
PQN_stats = Dict{Symbol, DataFrame}()
for key in PQN_keys
  PQN_stats[key] = stats[key]
end 
path_result_PQN = path_result*"PQN/"
p_iter = SolverBenchmark.performance_profile(PQN_stats, df -> cost_iter(df); legend=:bottomright)
savefig(p_iter, path_result_PQN*"iter_profile.pdf")
p_time = SolverBenchmark.performance_profile(PQN_stats, df -> cost_time(df); legend=:bottomright)
savefig(p_time, path_result_PQN*"time_profile.pdf")

# Limited-memory partitioned quasi-Newton profiles
PLQN_keys = [:PLSR1, :PLSE, :PLBFGS]
PLQN_stats = Dict{Symbol, DataFrame}()
for key in PLQN_keys
  PLQN_stats[key] = stats[key]
end 
path_result_PLQN = path_result*"PLQN/"
p_iter = SolverBenchmark.performance_profile(PLQN_stats, df -> cost_iter(df); legend=:bottomright)
savefig(p_iter, path_result_PLQN*"iter_profile.pdf")
p_time = SolverBenchmark.performance_profile(PLQN_stats, df -> cost_time(df); legend=:bottomright)
savefig(p_time, path_result_PLQN*"time_profile.pdf")



# Final profiles
final_keys =  [:PHv, :LBFGS, :PSR1, :PSE, :PLSE]
final_stats = Dict{Symbol, DataFrame}()
for key in final_keys
  final_stats[key] = stats[key]
end 
path_result_final = path_result*"final/"
p_iter = SolverBenchmark.performance_profile(final_stats, df -> cost_iter(df); legend=:bottomright)
savefig(p_iter, path_result_final*"iter_profile.pdf")
p_time = SolverBenchmark.performance_profile(final_stats, df -> cost_time(df); legend=:bottomright)
savefig(p_time, path_result_final*"time_profile.pdf")

