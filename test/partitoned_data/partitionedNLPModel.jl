
@testset "test PQNNLPModel (PBFGS, PLBFGS) MathOptNLPModel" begin
  n = 10
  nlp = MathOptNLPModel(OptimizationProblems.arwhead(n), name = "arwhead " * string(n))
  x = rand(n)

  pbfgsnlp = PQNNLPModel(nlp; name = :pbfgs)
  plbfgsnlp = PQNNLPModel(nlp; name = :plbfgs)

  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(pbfgsnlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plbfgsnlp, x)

  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(pbfgsnlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plbfgsnlp, x)
end
