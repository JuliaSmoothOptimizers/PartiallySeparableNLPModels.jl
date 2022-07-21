using PkgBenchmark, BenchmarkTools
using SolverBenchmark
using Plots

# perform obj benchmarks
results_obj = PkgBenchmark.benchmarkpkg(
  "PartiallySeparableNLPModels",
  script = "benchmark/obj.jl",
)

# process benchmark results and post gist
ENV["GKSwstype"] = 100
p = profile_solvers(results_obj)
savefig(p, "benchmark/results/profile_obj.pdf")


# perform grad benchmarks
results_grad = PkgBenchmark.benchmarkpkg(
  "PartiallySeparableNLPModels",
  script = "benchmark/grad.jl",
)

# process benchmark results and post gist
p = profile_solvers(results_grad)
savefig(p, "benchmark/results/profile_grad.pdf")
