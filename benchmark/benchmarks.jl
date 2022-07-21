using BenchmarkTools
using SolverBenchmark
using NLPModels
using ADNLPModels, NLPModelsJuMP
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using PartiallySeparableNLPModels

const SUITE = BenchmarkGroup()

ns = [10, 100, 1000]

SUITE["PSNLPS"] = BenchmarkGroup()
# SUITE["PSNLPS"]["OBJ"] = BenchmarkGroup()
# SUITE["PSNLPS"]["GRAD"] = BenchmarkGroup()

for n in ns

  nlp = arwhead(; n)#calcul de la fonction objectif
  name = nlp.meta.name
  x = rand(n)
  psnlp = PartiallySeparableNLPModel(nlp)

  SUITE["PSNLPS"]["OBJ, $name $n"] =
    @benchmarkable NLPModels.obj($psnlp, $x)

  g = similar(x)
  SUITE["PSNLPS"]["GRAD, $name $n"] =
  @benchmarkable NLPModels.grad!($psnlp, $x, $g)

end