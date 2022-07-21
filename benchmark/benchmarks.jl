using BenchmarkTools
using SolverBenchmark
using NLPModels
using ADNLPModels, NLPModelsJuMP
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using PartiallySeparableNLPModels

const SUITE = BenchmarkGroup()

ns = [10, 100, 1000]
nlps = map(n -> arwhead(; n), ns)

SUITE["OBJ"] = BenchmarkGroup()
SUITE["GRAD"] = BenchmarkGroup()

for n in ns

  nlp = arwhead(; n)#calcul de la fonction objectif
  name = nlp.meta.name
  x = rand(n)
  psnlp = PartiallySeparableNLPModel(nlp)

  SUITE["OBJ"]["$name $n"] =
    @benchmarkable NLPModels.obj($psnlp, $x)

  g = similar(x)
  SUITE["GRAD"]["$name $n"] =
  @benchmarkable NLPModels.grad!($psnlp, $x, $g)

end