# using Revise
using JSOSolvers
using NLPModels, ADNLPModels, NLPModelsModifiers
using PartiallySeparableNLPModels 
using Plots

include("function.jl")

# warm-up
n = 36
adnlp1 = ADNLPModel(limit, start_limit(n))
adnlp2 = ADNLPModel(limit, start_limit(n))
adnlp3 = ADNLPModel(limit, start_limit(n))

lbfgsnlp = LBFGSModel(adnlp1)
psr1nlp = PSR1NLPModel(adnlp2; merging=true)
plsenlp = PLSENLPModel(adnlp3; merging=true)

lbfgs_ges = trunk(lbfgsnlp; verbose=1)
psr1_ges = trunk(psr1nlp; verbose=1)
plse_ges = trunk(plsenlp; verbose=1)

# [lbfgs_ges.iter, psr1_ges.iter, plse_ges.iter]
# [lbfgs_ges.elapsed_time, psr1_ges.elapsed_time, plse_ges.elapsed_time]


lbfgs_iter_res = []
psr1_iter_res = []
plse_iter_res = []
lbfgs_time_res = []
psr1_time_res = []
plse_time_res = []
lbfgs_hprod_res = []
psr1_hprod_res = []
plse_hprod_res = []

lbfgs_time = true
psr1_time = true
plse_time = true

max_time = 9000.
max_eval = 50000

time_begining = time()
squared_nmin = 6
squared_nmax = 100

range = (len -> len^2).(collect(squared_nmin:squared_nmax))
for n in range 
  println("n: ", n, ", lbfgs, psr1, plse: ", lbfgs_time, " | ", psr1_time, " | ", plse_time)
  adnlp1 = ADNLPModel(limit, start_limit(n))
  adnlp2 = ADNLPModel(limit, start_limit(n))
  adnlp3 = ADNLPModel(limit, start_limit(n))  

  lbfgs_time && (lbfgsnlp = LBFGSModel(adnlp1))
  psr1_time && (psr1nlp = PSR1NLPModel(adnlp2; merging=true))
  plse_time && (plsenlp = PLSENLPModel(adnlp3; merging=true))

  lbfgs_time && (lbfgs_ges = trunk(lbfgsnlp; max_time, max_eval))
  psr1_time && (psr1_ges = trunk(psr1nlp; max_time, max_eval))
  plse_time && (plse_ges = trunk(plsenlp; max_time, max_eval))

  lbfgs_time && (push!(lbfgs_iter_res, lbfgs_ges.iter))
  psr1_time && (push!(psr1_iter_res, psr1_ges.iter))
  plse_time && (push!(plse_iter_res, plse_ges.iter))

  lbfgs_time && (push!(lbfgs_time_res, lbfgs_ges.elapsed_time))
  psr1_time && (push!(psr1_time_res, psr1_ges.elapsed_time))
  plse_time && (push!(plse_time_res, plse_ges.elapsed_time))

  lbfgs_time && (push!(lbfgs_hprod_res, lbfgsnlp.op.nprod))
  psr1_time && (push!(psr1_hprod_res, psr1nlp.model.counters.neval_hprod))
  plse_time && (push!(plse_hprod_res, plsenlp.model.counters.neval_hprod))

  @show [lbfgs_ges.elapsed_time, psr1_ges.elapsed_time, plse_ges.elapsed_time]
  @show [lbfgs_ges.iter, psr1_ges.iter, plse_ges.iter]
  @show [lbfgsnlp.op.nprod, psr1nlp.model.counters.neval_hprod, plsenlp.model.counters.neval_hprod]  
end

ENV["GKSwstype"]=100

path_result = pwd()*"/script/limits/results/"

# saving arrays
io = open(path_result*"iter.jl", "w+")
write(io, "lbfgs_iter_res = ")
write(io, string(lbfgs_iter_res))
write(io, "\n")
write(io, "psr1_iter_res = ")
write(io, string(psr1_iter_res))
write(io, "\n")
write(io, "plse_iter_res = ")
write(io, string(plse_iter_res))
write(io, "\n")
close(io)

io = open(path_result*"time.jl", "w+")
write(io, "lbfgs_time_res = ")
write(io, string(lbfgs_time_res))
write(io, "\n")
write(io, "psr1_time_res = ")
write(io, string(psr1_time_res))
write(io, "\n")
write(io, "plse_time_res = ")
write(io, string(plse_time_res))
write(io, "\n")
close(io)

io = open(path_result*"hprod.jl", "w+")
write(io, "lbfgs_hprod_res = ")
write(io, string(lbfgs_hprod_res))
write(io, "\n")
write(io, "psr1_hprod_res = ")
write(io, string(psr1_hprod_res))
write(io, "\n")
write(io, "plse_hprod_res = ")
write(io, string(plse_hprod_res))
write(io, "\n")
close(io)

# printing curves
p_iter = plot(range[1:length(lbfgs_iter_res)],[lbfgs_iter_res], xlabel = "n", ylabel="iterations", label = "LBFGS", lw = 3, legend=:topleft)
plot!(range[1:length(psr1_iter_res)], [psr1_iter_res], label = "PSR1", lw = 3)
plot!(range[1:length(plse_iter_res)], [plse_iter_res], label = "PLSE", lw = 3)
savefig(p_iter, path_result*"iter.pdf")


p_time = plot(range[1:length(lbfgs_time_res)],[lbfgs_time_res], xlabel = "n", ylabel="time (s)", label = "LBFGS", lw = 3, legend=:topleft)
plot!(range[1:length(psr1_time_res)], [psr1_time_res], label = "PSR1", lw = 3)
plot!(range[1:length(plse_time_res)], [plse_time_res], label = "PLSE", lw = 3)
savefig(p_time, path_result*"time.pdf")


p_time = plot(range[1:length(lbfgs_hprod_res)],[lbfgs_hprod_res], xlabel = "n", ylabel="number of Bₖv performed", label = "LBFGS", lw = 3, legend=:topleft)
plot!(range[1:length(psr1_hprod_res)], [psr1_hprod_res], label = "PSR1", lw = 3)
plot!(range[1:length(plse_hprod_res)], [plse_hprod_res], label = "PLSE", lw = 3)
savefig(p_time, path_result*"hprod.pdf")

time_ending = time()
time_duration = time_ending - time_begining