
@testset "test PartiallySeparableNLPModels (ADNLPModel)" begin
  n = 10
  nlp = ADNLPProblems.arwhead(; n)
  x = ones(n)

  pbfgsnlp = PBFGSNLPModel(nlp)
  pcsnlp = PCSNLPModel(nlp)
  plbfgsnlp = PLBFGSNLPModel(nlp)
  plsr1nlp = PLSR1NLPModel(nlp)
  plsenlp = PLSENLPModel(nlp)
  psr1nlp = PSR1NLPModel(nlp)
  psenlp = PSENLPModel(nlp)
  psnlp = PSNLPModel(nlp)

  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(pbfgsnlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(pcsnlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plbfgsnlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plsr1nlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plsenlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(psr1nlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(psenlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(psnlp, x)

  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(pbfgsnlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(pcsnlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plbfgsnlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plsr1nlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plsenlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(psr1nlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(psenlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(psnlp, x)

  v = [i % 2 == 0 ? 1.0 : 0.0 for i = 1:n]
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(pbfgsnlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(pcsnlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(plbfgsnlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(plsr1nlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(plsenlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(psr1nlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(psenlp, x, v)
  @test NLPModels.hprod(nlp, x, v) ≈ NLPModels.hprod(psnlp, x, v)
  
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(pcsnlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(plbfgsnlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(plsr1nlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(plsenlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(psr1nlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(psenlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(nlp, x, v; obj_weight = 1.5) ≈
        NLPModels.hprod(psnlp, x, v; obj_weight = 1.5)


  s = (si -> 0.5 * si).(ones(n))
  
  B_pbfgsnlp = update_nlp(pbfgsnlp, x, s; verbose=false)
  B_pcsnlp = update_nlp(pcsnlp, x, s; verbose=false)
  B_plbfgsnlp = update_nlp(plbfgsnlp, x, s; verbose=false)
  B_plsr1nlp = update_nlp(plsr1nlp, x, s; verbose=false)
  B_plsenlp = update_nlp(plsenlp, x, s; verbose=false)
  B_psr1nlp = update_nlp(psr1nlp, x, s; verbose=false)
  B_psenlp = update_nlp(psenlp, x, s; verbose=false)
  
  py = get_py(pbfgsnlp)
  @test py == get_py(pcsnlp)
  @test py == get_py(plbfgsnlp)
  @test py == get_py(plsr1nlp)
  @test py == get_py(plsenlp)
  @test py == get_py(psr1nlp)
  @test py == get_py(psenlp)
  build_v!(py)
  y = PartitionedStructures.get_v(py)
end

@testset "test PartiallySeparableNLPModels (MathOptNLPModel)" begin
  n = 10
  jump_model = PureJuMP.arwhead(; n)
  nlp = MathOptNLPModel(jump_model)
  x = ones(n)

  pbfgsnlp = PBFGSNLPModel(nlp)
  pcsnlp = PCSNLPModel(nlp)
  plbfgsnlp =PLBFGSNLPModel(nlp)
  plsr1nlp = PLSR1NLPModel(nlp)
  plsenlp = PLSENLPModel(nlp)
  psr1nlp = PSR1NLPModel(nlp)
  psenlp = PSENLPModel(nlp)
  psnlp = PSNLPModel(nlp)

  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(pbfgsnlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(pcsnlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plbfgsnlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plsr1nlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plsenlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(psr1nlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(psenlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(psnlp, x)

  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(pbfgsnlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(pcsnlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plbfgsnlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plsr1nlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plsenlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(psr1nlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(psenlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(psnlp, x)


  v = [i % 2 == 0 ? 1.0 : 0.0 for i = 1:n]
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(pbfgsnlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(pcsnlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(plbfgsnlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(plsr1nlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(plsenlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(psr1nlp, x, v)
  @test NLPModels.hprod(pbfgsnlp, x, v) == NLPModels.hprod(psenlp, x, v)
  @test NLPModels.hprod(nlp, x, v) ≈ NLPModels.hprod(psnlp, x, v)
  
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(pcsnlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(plbfgsnlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(plsr1nlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(plsenlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(psr1nlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, x, v; obj_weight = 1.5) == NLPModels.hprod(psenlp, x, v; obj_weight = 1.5)
  @test NLPModels.hprod(nlp, x, v; obj_weight = 1.5) ≈
        NLPModels.hprod(psnlp, x, v; obj_weight = 1.5)


  s = (si -> 0.5 * si).(ones(n))
  
  B_pbfgsnlp = update_nlp(pbfgsnlp, x, s; verbose=false)
  B_pcsnlp = update_nlp(pcsnlp, x, s; verbose=false)
  B_plbfgsnlp = update_nlp(plbfgsnlp, x, s; verbose=false)
  B_plsr1nlp = update_nlp(plsr1nlp, x, s; verbose=false)
  B_plsenlp = update_nlp(plsenlp, x, s; verbose=false)
  B_psr1nlp = update_nlp(psr1nlp, x, s; verbose=false)
  B_psenlp = update_nlp(psenlp, x, s; verbose=false)
  
  py = get_py(pbfgsnlp)
  @test py == get_py(pcsnlp)
  @test py == get_py(plbfgsnlp)
  @test py == get_py(plsr1nlp)
  @test py == get_py(plsenlp)
  @test py == get_py(psr1nlp)
  @test py == get_py(psenlp)
  build_v!(py)
  y = PartitionedStructures.get_v(py)
end

@testset "Partitioned LinearOperators" begin
  n = 10
  nlp = ADNLPProblems.arwhead(; n)
  x0 = nlp.meta.x0
  v = ones(n)

  pbfgsnlp = PBFGSNLPModel(nlp)
  pcsnlp = PCSNLPModel(nlp)
  plbfgsnlp = PLBFGSNLPModel(nlp)
  plsr1nlp = PLSR1NLPModel(nlp)
  plsenlp = PLSENLPModel(nlp)
  psr1nlp = PSR1NLPModel(nlp)
  psenlp = PSENLPModel(nlp)

  B_pbfgs = LinearOperator(pbfgsnlp)
  B_pcs = LinearOperator(pcsnlp)
  B_plbfgs = LinearOperator(plbfgsnlp)
  B_plsr1 = LinearOperator(plsr1nlp)
  B_plse = LinearOperator(plsenlp)
  B_psr1 = LinearOperator(psr1nlp)


  B_pbfgsv = B_pbfgs * v
  @test B_pbfgsv == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 9.0]
  B_pcsv = B_pcs * v
  @test B_pcsv == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 9.0]
  B_plbfgsv = B_plbfgs * v
  @test B_plbfgsv == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 9.0]
  B_plsr1v = B_plsr1 * v
  @test B_plsr1v == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 9.0]
  B_plsev = B_plse * v
  @test B_plsev == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 9.0]
  B_psr1v = B_psr1 * v
  @test B_psr1v == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 9.0]

  epm_bfgs = get_pB(pbfgsnlp)
  # epm_pcsnlp = get_pB(pcsnlp)
  # epm_plbfgsnlp = get_pB(plbfgsnlp)
  # epm_plsr1nlp = get_pB(plsr1nlp)
  # epm_plsenlp = get_pB(plsenlp)
  # epm_psr1nlp = get_pB(psr1nlp)

  # epv = epv_from_epm(epm_bfgs)
  
  update_nlp!(pbfgsnlp, x0, ones(n); verbose = false)
  update_nlp!(pcsnlp, x0, ones(n); verbose = false)
  update_nlp!(plbfgsnlp, x0, ones(n); verbose = false)
  update_nlp!(plsr1nlp, x0, ones(n); verbose = false)
  update_nlp!(plsenlp, x0, ones(n); verbose = false)
  update_nlp!(psr1nlp, x0, ones(n); verbose = false)
  update_nlp!(psenlp, x0, ones(n); verbose = false)

  @test B_pbfgs * v != B_pbfgsv
  @test B_pbfgs * v != B_pbfgsv
  @test B_pcs * v != B_pcsv
  @test B_plbfgs * v != B_plbfgsv
  # @test B_plsr1 * v != B_plsr1v
  @test B_plse * v != B_plsev
  @test B_psr1 * v != B_psr1v

end

@testset "show" begin
  n = 10
  nlp = ADNLPProblems.arwhead(; n)

  psnlp = PSNLPModel(nlp)
  res = show(psnlp)

  @test res == nothing
end
