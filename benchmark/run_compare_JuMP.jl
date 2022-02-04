using PkgBenchmark
using SolverBenchmark
using Plots

# perform benchmarks
results = PkgBenchmark.benchmarkpkg("PartiallySeparableNLPModels", script="benchmark/compare_with_JuMP.jl")

# process benchmark results and post gist
ENV["GKSwstype"]=100
p = profile_solvers(results)
savefig(p, "benchmark/profile_JuMP.pdf")
