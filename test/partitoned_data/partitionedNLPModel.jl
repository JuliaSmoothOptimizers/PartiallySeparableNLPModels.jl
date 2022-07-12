
@testset "test PQNNLPModel (PBFGS, PLBFGS) MathOptNLPModel" begin
  n = 10
  nlp = MathOptNLPModel(OptimizationProblems.arwhead(n), name = "arwhead " * string(n))
  x = ones(n)

  pbfgsnlp = PQNNLPModel(nlp; name = :pbfgs)
  plbfgsnlp = PQNNLPModel(nlp; name = :plbfgs)

  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(pbfgsnlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plbfgsnlp, x)

  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(pbfgsnlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plbfgsnlp, x)

  s = (si -> 0.5*si).(ones(n))
  B_pbfgs = update_nlp(pbfgsnlp, x, s)
  B_plbfgs = update_nlp(plbfgsnlp, x, s)
  py = get_py(pbfgsnlp.part_data)
  build_v!(py)
  y = get_v(py)

end
