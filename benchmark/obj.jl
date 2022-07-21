using BenchmarkTools
using JuMP, MathOptInterface
using PkgBenchmark

using PartiallySeparableNLPModels
using ExpressionTreeForge
using NLPModels, NLPModelsJuMP, ADNLPModels
using ForwardDiff, ReverseDiff
using OptimizationProblems, OptimizationProblems.PureJuMP


function woods_adnlp(; n::Int = default_nvar, type::Val{T} = Val(Float64), kwargs...) where {T}
  (n % 4 == 0) || @warn("woods: number of variables adjusted to be a multiple of 4")
  n = 4 * max(1, div(n, 4))
  function f(x)
    n = length(x)
    return sum(
      100 * (x[4 * i - 2] - x[4 * i - 3]^2)^2 +
      (1 - x[4 * i - 3])^2 +
      90 * (x[4 * i] - x[4 * i - 1]^2)^2 +
      (1 - x[4 * i - 1])^2 +
      10 * (x[4 * i - 2] + x[4 * i] - 2)^2 +
      T(0.1) * (x[4 * i - 2] - x[4 * i])^2 for i = 1:div(n, 4)
    )
  end

  x0 = -3 * ones(T, n)
  x0[2 * (collect(1:div(n, 2)))] .= -one(T)

  return ADNLPModels.ADNLPModel(f, x0, name = "woods"; kwargs...)
end

const SUITE = BenchmarkGroup()

function models(n::Int)
  model = woods(;n=n)
  evaluator = JuMP.NLPEvaluator(model)
  MathOptInterface.initialize(evaluator, [:ExprGraph, :Hess])
  return (model, evaluator)
end

# problems = models.(n_var)
n_var = [10, 50, 100]

function run_comparaison(n_var)  
  for n_not_rounded in n_var

    (model_jump, evaluator) = models(n_not_rounded)

    nlp = MathOptNLPModel(model_jump)
    n = nlp.meta.nvar
    adnlp_forward = woods_adnlp(;n, adbackend=ADNLPModels.ZygoteAD)
    # adnlp_zygote = woods_adnlp(;n, adbackend=ADNLPModels.ZygoteAD)
    adnlp_reverse = woods_adnlp(;n, adbackend=ADNLPModels.ReverseDiffAD)
    psnlp = PartiallySeparableNLPModel(nlp)

    name = nlp.meta.name
    x = ones(n)
    y = ones(n)

    SUITE["PSNLP"] = BenchmarkGroup()
    SUITE["JuMP"] = BenchmarkGroup()
    SUITE["ADNLP_Forward"] = BenchmarkGroup()
    SUITE["ADNLP_Reverse"] = BenchmarkGroup()
    # SUITE["ADNLP_Zygote"] = BenchmarkGroup()
    
    SUITE["PSNLP"]["OBJ"] = BenchmarkGroup()    
    SUITE["JuMP"]["OBJ"] = BenchmarkGroup()
    SUITE["ADNLP_Forward"]["OBJ"] = BenchmarkGroup()
    SUITE["ADNLP_Reverse"]["OBJ"] = BenchmarkGroup()
    # SUITE["ADNLP_Zygote"]["OBJ"] = BenchmarkGroup()

    
    # Objective
    SUITE["PSNLP"]["OBJ"]["$name $n"] =
      @benchmarkable NLPModels.obj($psnlp, $x)
    SUITE["JuMP"]["OBJ"]["$name $n"] =
      @benchmarkable MathOptInterface.eval_objective($evaluator, $x)
    SUITE["ADNLP_Forward"]["OBJ"]["$name $n"] =
      @benchmarkable NLPModels.obj($adnlp_forward, $x)
    SUITE["ADNLP_Reverse"]["OBJ"]["$name $n"] =
      @benchmarkable NLPModels.obj($adnlp_reverse, $x)
    # SUITE["ADNLP_Zygote"]["OBJ"]["$name $n"] =
    #   @benchmarkable NLPModels.obj($adnlp_zygote, $x)
  end
end

run_comparaison(n_var)