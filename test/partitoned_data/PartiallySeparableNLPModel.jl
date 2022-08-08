
@testset "test PartiallySeparableNLPModel (PBFGS, PLBFGS) ADNLPModel" begin
  n = 10
  nlp = ADNLPProblems.arwhead(; n)

  x = ones(n)

  pbfgsnlp = PartiallySeparableNLPModel(nlp; name = :pbfgs)
  plbfgsnlp = PartiallySeparableNLPModel(nlp; name = :plbfgs)

  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(pbfgsnlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plbfgsnlp, x)

  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(pbfgsnlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plbfgsnlp, x)

  v = [i % 2 == 0 ? 1.0 : 0.0 for i = 1:n]
  @test NLPModels.hprod(nlp, x, v) ≈ NLPModels.hprod(pbfgsnlp, x, v)
  @test NLPModels.hprod(nlp, x, v) ≈ NLPModels.hprod(plbfgsnlp, x, v)

  @test NLPModels.hprod(nlp, x, v; obj_weight = 1.5) ≈
        NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(nlp, x, v; obj_weight = 1.5) ≈
        NLPModels.hprod(plbfgsnlp, x, v; obj_weight = 1.5)

  s = (si -> 0.5 * si).(ones(n))
  B_pbfgs = update_nlp(pbfgsnlp, x, s)
  B_plbfgs = update_nlp(plbfgsnlp, x, s)
  py = get_py(pbfgsnlp.part_data)
  build_v!(py)
  y = get_v(py)
end

@testset "test PartiallySeparableNLPModel (PBFGS, PLBFGS) MathOptNLPModel" begin
  n = 10
  jump_model = PureJuMP.arwhead(; n)
  nlp = MathOptNLPModel(jump_model)
  x = ones(n)

  pbfgsnlp = PartiallySeparableNLPModel(nlp; name = :pbfgs)
  plbfgsnlp = PartiallySeparableNLPModel(nlp; name = :plbfgs)

  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(pbfgsnlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plbfgsnlp, x)

  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(pbfgsnlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plbfgsnlp, x)

  v = [i % 2 == 0 ? 1.0 : 0.0 for i = 1:n]
  @test NLPModels.hprod(nlp, x, v) ≈ NLPModels.hprod(pbfgsnlp, x, v)
  @test NLPModels.hprod(nlp, x, v) ≈ NLPModels.hprod(plbfgsnlp, x, v)

  @test NLPModels.hprod(nlp, x, v; obj_weight = 1.5) ≈
        NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(nlp, x, v; obj_weight = 1.5) ≈
        NLPModels.hprod(plbfgsnlp, x, v; obj_weight = 1.5)

  s = (si -> 0.5 * si).(ones(n))
  B_pbfgs = update_nlp(pbfgsnlp, x, s)
  B_plbfgs = update_nlp(plbfgsnlp, x, s)
  py = get_py(pbfgsnlp.part_data)
  build_v!(py)
  y = get_v(py)
end

@testset "show" begin
  n = 10
  nlp = ADNLPProblems.arwhead(; n)

  pbfgsnlp = PartiallySeparableNLPModel(nlp; name = :pbfgs)
  res = show(pbfgsnlp)

  @test res == nothing
end