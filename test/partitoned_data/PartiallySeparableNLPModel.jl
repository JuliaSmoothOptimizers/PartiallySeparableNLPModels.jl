
@testset "test PartiallySeparableNLPModel (PBFGS, PLBFGS) MathOptNLPModel" begin
  n = 10
  nlp = arwhead(; n)

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


@testset "Partitioned LinearOperators" begin
  n = 10
  nlp = arwhead(; n)

  v = ones(n)

  pbfgsnlp = PartiallySeparableNLPModel(nlp; name = :pbfgs)
  plsesnlp = PartiallySeparableNLPModel(nlp; name = :plse)

  B_pbfgs = LinearOperator(pbfgsnlp)
  B_plse = LinearOperator(pbfgsnlp)

  B_pbfgsv = B_pbfgs*v
  @test B_pbfgsv == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 9.0]

  B_plsev = B_plse*v
  @test B_plsev == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 9.0]


  part_data_pbfgs = pbfgsnlp.part_data
  epm_bfgs = get_pB(part_data_pbfgs)
  epv = epv_from_epm(epm_bfgs)
  update!(epm_bfgs, epv, ones(n); name=:pbfgs, verbose=false)
  @test B_pbfgs*v != B_pbfgsv


  part_data_plse = plsesnlp.part_data
  epm_lse = get_pB(part_data_plse)
  update!(epm_lse, epv, ones(n), verbose=false)
  @test B_plse*v != B_plsev

end 