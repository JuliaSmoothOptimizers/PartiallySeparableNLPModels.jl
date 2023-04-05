@testset "test PartiallySeparableNLPModels (ADNLPModel)" begin
  n = 10
  nlp = ADNLPProblems.arwhead(; n)

  pbfgsnlp = PBFGSNLPModel(nlp)
  pcsnlp = PCSNLPModel(nlp)
  plbfgsnlp = PLBFGSNLPModel(nlp)
  plsr1nlp = PLSR1NLPModel(nlp)
  plsenlp = PLSENLPModel(nlp)
  psr1nlp = PSR1NLPModel(nlp)
  psenlp = PSENLPModel(nlp)
  psnlp = PSNLPModel(nlp)

  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(pbfgsnlp, pbfgsnlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(pcsnlp, pcsnlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(plbfgsnlp, plbfgsnlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(plsr1nlp, plsr1nlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(plsenlp, plsenlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(psr1nlp, psr1nlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(psenlp, psenlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(psnlp, psnlp.meta.x0)

  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(pbfgsnlp, pbfgsnlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(pcsnlp, pcsnlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(plbfgsnlp, plbfgsnlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(plsr1nlp, plsr1nlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(plsenlp, plsenlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(psr1nlp, psr1nlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(psenlp, psenlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(psnlp, psnlp.meta.x0))

  v = ones(n)
  pv = similar(pbfgsnlp.meta.x0)
  pv .= 1.0
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(pcsnlp, pcsnlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(plbfgsnlp, plbfgsnlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(plsr1nlp, plsr1nlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(plsenlp, plsenlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(psr1nlp, psr1nlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(psenlp, psenlp.meta.x0, pv)
  @test NLPModels.hprod(nlp, nlp.meta.x0, v) ≈ Vector(NLPModels.hprod(psnlp, psnlp.meta.x0, pv))

  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(pcsnlp, pcsnlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(plbfgsnlp, plbfgsnlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(plsr1nlp, plsr1nlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(plsenlp, plsenlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(psr1nlp, psr1nlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(psenlp, psenlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(nlp, nlp.meta.x0, v; obj_weight = 1.5) ≈
        Vector(NLPModels.hprod(psnlp, psnlp.meta.x0, pv; obj_weight = 1.5))
end

@testset "test PartiallySeparableNLPModels (JuMPModel)" begin
  n = 10
  jump_model = PureJuMP.arwhead(; n)
  nlp = MathOptNLPModel(jump_model)

  pbfgsnlp = PBFGSNLPModel(nlp)
  pcsnlp = PCSNLPModel(nlp)
  plbfgsnlp = PLBFGSNLPModel(nlp)
  plsr1nlp = PLSR1NLPModel(nlp)
  plsenlp = PLSENLPModel(nlp)
  psr1nlp = PSR1NLPModel(nlp)
  psenlp = PSENLPModel(nlp)
  psnlp = PSNLPModel(nlp)

  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(pbfgsnlp, pbfgsnlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(pcsnlp, pcsnlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(plbfgsnlp, plbfgsnlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(plsr1nlp, plsr1nlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(plsenlp, plsenlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(psr1nlp, psr1nlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(psenlp, psenlp.meta.x0)
  @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(psnlp, psnlp.meta.x0)

  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(pbfgsnlp, pbfgsnlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(pcsnlp, pcsnlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(plbfgsnlp, plbfgsnlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(plsr1nlp, plsr1nlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(plsenlp, plsenlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(psr1nlp, psr1nlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(psenlp, psenlp.meta.x0))
  @test NLPModels.grad(nlp, nlp.meta.x0) ≈ Vector(NLPModels.grad(psnlp, psnlp.meta.x0))

  v = ones(n)
  pv = similar(pbfgsnlp.meta.x0)
  pv .= 1.0
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(pcsnlp, pcsnlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(plbfgsnlp, plbfgsnlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(plsr1nlp, plsr1nlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(plsenlp, plsenlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(psr1nlp, psr1nlp.meta.x0, pv)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv) ==
        NLPModels.hprod(psenlp, psenlp.meta.x0, pv)
  @test NLPModels.hprod(nlp, nlp.meta.x0, v) ≈ Vector(NLPModels.hprod(psnlp, psnlp.meta.x0, pv))

  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(pcsnlp, pcsnlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(plbfgsnlp, plbfgsnlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(plsr1nlp, plsr1nlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(plsenlp, plsenlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(psr1nlp, psr1nlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(pbfgsnlp, pbfgsnlp.meta.x0, pv; obj_weight = 1.5) ==
        NLPModels.hprod(psenlp, psenlp.meta.x0, pv; obj_weight = 1.5)
  @test NLPModels.hprod(nlp, nlp.meta.x0, v; obj_weight = 1.5) ≈
        Vector(NLPModels.hprod(psnlp, psnlp.meta.x0, pv; obj_weight = 1.5))

  @testset "Backend general tests" begin
    pbfgsnlp_moiobj = PBFGSNLPModel(nlp, objectivebackend = :moiobj)
    @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(pbfgsnlp_moiobj, pbfgsnlp_moiobj.meta.x0)

    pbfgsnlp_moielt = PBFGSNLPModel(nlp, objectivebackend = :moielt, gradientbackend = :moielt)
    @test NLPModels.grad(pbfgsnlp, pbfgsnlp.meta.x0) ≈
          NLPModels.grad(pbfgsnlp_moielt, pbfgsnlp_moielt.meta.x0)
    @test NLPModels.obj(pbfgsnlp, pbfgsnlp.meta.x0) ≈
          NLPModels.obj(pbfgsnlp_moielt, pbfgsnlp_moielt.meta.x0)

    pbfgsnlp_modifiedmoiobj =
      PBFGSNLPModel(nlp, objectivebackend = :modifiedmoiobj, gradientbackend = :modifiedmoiobj)
    @test NLPModels.obj(pbfgsnlp, pbfgsnlp.meta.x0) ≈
          NLPModels.obj(pbfgsnlp_modifiedmoiobj, pbfgsnlp_modifiedmoiobj.meta.x0)
    @test NLPModels.grad(pbfgsnlp, pbfgsnlp.meta.x0) ≈
          NLPModels.grad(pbfgsnlp_modifiedmoiobj, pbfgsnlp_modifiedmoiobj.meta.x0)

    pbfgsnlp_spjac = PBFGSNLPModel(nlp, objectivebackend = :spjacmoi, gradientbackend = :spjacmoi)
    @test NLPModels.obj(nlp, nlp.meta.x0) ≈ NLPModels.obj(pbfgsnlp_spjac, pbfgsnlp_spjac.meta.x0)
    @test NLPModels.grad(pbfgsnlp, pbfgsnlp.meta.x0) ≈
          NLPModels.grad(pbfgsnlp_spjac, pbfgsnlp_spjac.meta.x0)

    n = length(nlp.meta.x0)
    x = rand(n)
    px = similar(pbfgsnlp.meta.x0)
    PartitionedVectors.set!(px, x)

    @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(pbfgsnlp_moiobj, px)
    @test NLPModels.obj(pbfgsnlp, px) ≈ NLPModels.obj(pbfgsnlp_moielt, px)
    @test NLPModels.grad(pbfgsnlp, px) ≈ NLPModels.grad(pbfgsnlp_moielt, px)
    @test NLPModels.obj(pbfgsnlp, px) ≈ NLPModels.obj(pbfgsnlp_modifiedmoiobj, px)
    @test NLPModels.grad(pbfgsnlp, px) ≈ NLPModels.grad(pbfgsnlp_modifiedmoiobj, px)
    @test NLPModels.obj(pbfgsnlp, px) ≈ NLPModels.obj(pbfgsnlp_spjac, px)
    @test NLPModels.grad(pbfgsnlp, px) ≈ NLPModels.grad(pbfgsnlp_spjac, px)
  end
end

@testset "hessop" begin
  n = 10
  nlp = ADNLPProblems.arwhead(; n)

  pbfgsnlp = PBFGSNLPModel(nlp)
  pcsnlp = PCSNLPModel(nlp)
  plbfgsnlp = PLBFGSNLPModel(nlp)
  plsr1nlp = PLSR1NLPModel(nlp)
  plsenlp = PLSENLPModel(nlp)
  psr1nlp = PSR1NLPModel(nlp)
  psenlp = PSENLPModel(nlp)
  psnlp = PSNLPModel(nlp)

  x = psnlp.meta.x0
  v = similar(x)
  v .= 1

  op_pbfgs = NLPModels.hess_op(pbfgsnlp, x)
  op_pcs = NLPModels.hess_op(pcsnlp, x)
  op_plbfgs = NLPModels.hess_op(plbfgsnlp, x)
  op_plsr1 = NLPModels.hess_op(plsr1nlp, x)
  op_plse = NLPModels.hess_op(plsenlp, x)
  op_psr1 = NLPModels.hess_op(psr1nlp, x)
  op_ps = NLPModels.hess_op(psnlp, x)

  Hv = similar(v; simulate_vector = false)
  mul!(Hv, op_ps, v, 1, 0.0)

  x0 = pbfgsnlp.meta.x0
  ps = similar(x0)
  ps .= 1
  g = similar(x0; simulate_vector = false)
  g1 = similar(x0; simulate_vector = false)
  py = similar(x0; simulate_vector = false)

  NLPModels.grad!(psenlp, x0, g)
  NLPModels.grad!(psenlp, x0 + ps, g1)

  py .= g1 .- g

  push!(pbfgsnlp, ps, py)
  push!(pcsnlp, ps, py)
  push!(plbfgsnlp, ps, py)
  push!(plsr1nlp, ps, py)
  push!(plsenlp, ps, py)
  push!(psr1nlp, ps, py)
  push!(psenlp, ps, py)

  pbfgs_s = op_pbfgs * ps
  pcs_s = op_pcs * ps
  plbfgs_s = op_plbfgs * ps
  plsr1_s = op_plsr1 * ps
  plse_s = op_plse * ps
  psr1_s = op_psr1 * ps

  pbfgs_s.simulate_vector = false
  pcs_s.simulate_vector = false
  plbfgs_s.simulate_vector = false
  plsr1_s.simulate_vector = false
  plse_s.simulate_vector = false
  psr1_s.simulate_vector = false

  # They do not all satisfy the secant equation because not every elements are updated.
  # Limited-memory partitioned quasi-Newton operators rely on damped operators, making them not satisfy secant equation.
  @test isapprox(norm(Vector(pbfgs_s) - Vector(py)), 0, atol = 1e-10)
  @test isapprox(norm(Vector(pcs_s) - Vector(py)), 0, atol = 1e-10)
  @test isapprox(norm(Vector(plbfgs_s) - Vector(py)), 0, atol = 1e-10)
  # @test isapprox(norm(Vector(plsr1_s) - Vector(py)), 0, atol = 1e-10)  
  @test isapprox(norm(Vector(plse_s) - Vector(py)), 0, atol = 1e-10)
  @test isapprox(norm(Vector(psr1_s) - Vector(py)), 0, atol = 1e-10)

  @testset "reset data" begin
    NLPModels.reset_data!(pbfgsnlp)
    NLPModels.reset_data!(pcsnlp)
    NLPModels.reset_data!(plbfgsnlp)
    NLPModels.reset_data!(plsr1nlp)
    NLPModels.reset_data!(plsenlp)
    NLPModels.reset_data!(psr1nlp)
    NLPModels.reset_data!(psenlp)

    @test Matrix(pcsnlp.op) == Matrix(pbfgsnlp.op)
    @test Matrix(psr1nlp.op) == Matrix(pbfgsnlp.op)
    @test Matrix(psenlp.op) == Matrix(pbfgsnlp.op)
  end
end

@testset "show" begin
  n = 10
  nlp = ADNLPProblems.arwhead(; n)

  psnlp = PSNLPModel(nlp)
  res = show(psnlp)

  pqnnlp = PBFGSNLPModel(nlp)
  res = show(pqnnlp)

  @test res == nothing

  meta = psnlp.meta
  show(meta)
end

@testset "Backend errors" begin
  using PartiallySeparableNLPModels.PartitionedBackends

  mutable struct FakeBackend{T} <: PartitionedBackend{T}
  end

  fb = FakeBackend{Float64}()
  x = rand(5)
  g = similar(x)
  Hv = similar(x)

  @test_throws ErrorException objective(fb, x)
  @test_throws ErrorException partitioned_gradient!(fb, x, g)
  @test_throws ErrorException partitioned_hessian_prod!(fb, x, g, Hv)
end
