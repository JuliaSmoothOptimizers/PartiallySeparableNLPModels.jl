using PartiallySeparableNLPModels.ModAbstractPSNLPModels

@testset "PQN structure" begin
  n = 20
  adnlp = ADNLPProblems.rosenbrock(; n)

  n = adnlp.meta.nvar 

  x = (x -> 2 * x).(ones(n))
  y = (x -> 0.5 * x).(ones(n))

  pbfgsnlp = PBFGSNLPModel(adnlp)

  objx = evaluate_obj_part_data(pbfgsnlp, x)
  @test objx == NLPModels.obj(adnlp, x)

  objy = evaluate_obj_part_data(pbfgsnlp, y)
  @test objy ≈ NLPModels.obj(adnlp, y)

  gx = evaluate_grad_part_data(pbfgsnlp, x)
  @test NLPModels.grad(adnlp, x) == gx

  gy = evaluate_grad_part_data(pbfgsnlp, y)
  @test NLPModels.grad(adnlp, y) ≈ gy

  Bk = Matrix(pbfgsnlp.pB)

  x = (x -> 2 * x).(ones(n))
  s = (x -> 0.1 * x).(ones(n))

  update_nlp!(pbfgsnlp, x, s; verbose = false )
  Bk1 = Matrix(pbfgsnlp.pB)
  epv_y = pbfgsnlp.py
  PartitionedStructures.build_v!(epv_y)
  _y = PartitionedStructures.get_v(epv_y)
  @test isapprox(norm(Bk1 * s - _y), 0, atol = 1e-10)

  res = product_part_data_x(pbfgsnlp, x)
  @test res ≈ Bk1 * x
end

@testset "PartiallySeparableNLPModels, update_nlp!(psnlp, x, s)" begin
  n = 20
  adnlp = ADNLPProblems.rosenbrock(; n)
  n = adnlp.meta.nvar 

  x = (x -> 2 * x).(ones(n))
  s = (x -> 0.5 * x).(ones(n))

  pbfgsnlp = PBFGSNLPModel(adnlp)
  pcsnlp = PCSNLPModel(adnlp)
  plbfgsnlp = PLBFGSNLPModel(adnlp)
  plsr1nlp = PLSR1NLPModel(adnlp)
  plsenlp = PLSENLPModel(adnlp)
  psr1nlp = PSR1NLPModel(adnlp)
  psenlp = PSENLPModel(adnlp)

  update_nlp!(pbfgsnlp, x, s; verbose = false)
  update_nlp!(pcsnlp, x, s; verbose = false)
  update_nlp!(plbfgsnlp, x, s; verbose = false)
  update_nlp!(psenlp, x, s; verbose = false)
  update_nlp!(plsenlp, x, s; verbose = false)
  update_nlp!(psr1nlp, x, s; verbose = false)

  @test plbfgsnlp.py == pbfgsnlp.py
  @test plbfgsnlp.py == pcsnlp.py
  @test plbfgsnlp.py == psr1nlp.py
  @test plbfgsnlp.py == psenlp.py
  @test plbfgsnlp.py == plsenlp.py

  epv_y = plbfgsnlp.py
  PartitionedStructures.build_v!(epv_y)
  y = PartitionedStructures.get_v(epv_y)

  partitioned_matrix(nlp) = Matrix(nlp.pB)

  # in the case of the Rosenbrock equation, for the given x,s and induces y, every partitioned update ensure the secant equation.
  @test isapprox(norm(partitioned_matrix(pbfgsnlp) * s - y), 0, atol = 1e-10)  
  @test isapprox(norm(partitioned_matrix(pcsnlp) * s - y), 0, atol = 1e-10)
  @test isapprox(norm(partitioned_matrix(plbfgsnlp) * s - y), 0, atol = 1e-10)
  @test isapprox(norm(partitioned_matrix(psenlp) * s - y), 0, atol = 1e-10)
  @test isapprox(norm(partitioned_matrix(plsenlp) * s - y), 0, atol = 1e-10)
  @test isapprox(norm(partitioned_matrix(psr1nlp) * s - y), 0, atol = 1e-10)
end

@testset "methods" begin
  n = 20
  adnlp = ADNLPProblems.rosenbrock(; n)
  n = adnlp.meta.nvar 

  pbfgsnlp = PBFGSNLPModel(adnlp)

  x = (x -> 2 * x).(ones(n))
  s = (x -> 0.5 * x).(ones(n))

  @test get_vec_elt_fun(pbfgsnlp) == pbfgsnlp.vec_elt_fun
  @test get_vec_elt_complete_expr_tree(pbfgsnlp) == pbfgsnlp.vec_elt_complete_expr_tree
  @test get_element_expr_tree_table(pbfgsnlp) == pbfgsnlp.element_expr_tree_table
  @test get_vec_compiled_element_gradients(pbfgsnlp) == pbfgsnlp.vec_compiled_element_gradients

  set_s!(pbfgsnlp, s)
  @test get_s(pbfgsnlp) == s

  set_n!(pbfgsnlp, n + 1)
  @test get_n(pbfgsnlp) == n + 1
  set_n!(pbfgsnlp, n)

  N = get_N(pbfgsnlp)
  set_N!(pbfgsnlp, N + 1)
  @test get_N(pbfgsnlp) == N + 1
  set_N!(pbfgsnlp, N)

  v = rand(n)
  set_v!(pbfgsnlp, v)
  @test ModAbstractPSNLPModels.get_v(pbfgsnlp) == v

  onesn = ones(n)
  epv = similar(get_pv(pbfgsnlp))
  epv_from_v!(epv, onesn)

  set_pv!(pbfgsnlp, onesn)
  @test get_pv(pbfgsnlp) == epv

  set_pv!(pbfgsnlp, epv)
  @test get_pv(pbfgsnlp) == epv

  set_ps!(pbfgsnlp, onesn)
  @test get_ps(pbfgsnlp) == epv

  set_pg!(pbfgsnlp, onesn)
  @test get_pg(pbfgsnlp) == epv

  set_pg!(pbfgsnlp, epv)
  @test get_pg(pbfgsnlp) == epv

  set_py!(pbfgsnlp, onesn)
  @test get_py(pbfgsnlp) == epv

  set_phv!(pbfgsnlp, onesn)
  @test get_phv(pbfgsnlp) == epv

  epm = epm_from_epv(epv)

  set_pB!(pbfgsnlp, epm)
  @test get_pB(pbfgsnlp) == epm
end
