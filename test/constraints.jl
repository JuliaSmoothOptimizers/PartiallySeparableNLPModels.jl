@testset "constraints (ADNLPModel)" begin
  nlp = ADNLPProblems.hs6()
  psnlp = PSNLPModel(nlp)

  @test psnlp.meta.ncon == nlp.meta.ncon
  @test Vector(psnlp.meta.lcon) == nlp.meta.lcon
  @test Vector(psnlp.meta.ucon) == nlp.meta.ucon

  x = psnlp.meta.x0
  xv = Vector(x)

  c = NLPModels.cons(nlp, xv)
  pc = NLPModels.cons(psnlp, x)
  @test c ≈ Vector(pc)

  J = NLPModels.jac(nlp, xv)
  pJ = NLPModels.jac(psnlp, x)
  @test Matrix(J) ≈ Matrix(pJ)

  @testset "two constraints, overlapping variables" begin
    n = 3
    f(y) = y[1]^2 + y[2]^2 + y[3]^2
    function c!(cy, y)
      cy[1] = y[1] + y[2] - 1.0
      cy[2] = y[2] * y[3] - 2.0
      cy
    end
    nlp2 = ADNLPModel!(f, ones(n), c!, zeros(2), zeros(2))
    psnlp2 = PSNLPModel(nlp2)

    @test psnlp2.meta.ncon == 2
    x2 = psnlp2.meta.x0
    x2v = Vector(x2)

    @test NLPModels.cons(nlp2, x2v) ≈ Vector(NLPModels.cons(psnlp2, x2))
    @test Matrix(NLPModels.jac(nlp2, x2v)) ≈ Matrix(NLPModels.jac(psnlp2, x2))

    y = rand(n)
    py = similar(x2)
    PartitionedVectors.set!(py, y)
    @test NLPModels.cons(nlp2, y) ≈ Vector(NLPModels.cons(psnlp2, py))
    @test Matrix(NLPModels.jac(nlp2, y)) ≈ Matrix(NLPModels.jac(psnlp2, py))
  end
end

@testset "constraints (MathOptNLPModel, @NLconstraint)" begin
  jmodel = JuMP.Model()
  JuMP.@variable(jmodel, y[1:3])
  JuMP.@NLconstraint(jmodel, y[1] + y[2] - 1.0 == 0)
  JuMP.@NLconstraint(jmodel, y[2] * y[3] <= 2.0)
  JuMP.@NLobjective(jmodel, Min, sum(y[i]^2 for i = 1:3))
  nlp = MathOptNLPModel(jmodel)
  psnlp = PSNLPModel(nlp)

  @test psnlp.meta.ncon == nlp.meta.ncon == 2

  x = psnlp.meta.x0
  xv = Vector(x)

  @test NLPModels.cons(nlp, xv) ≈ Vector(NLPModels.cons(psnlp, x))
  @test Matrix(NLPModels.jac(nlp, xv)) ≈ Matrix(NLPModels.jac(psnlp, x))

  y = rand(3)
  py = similar(x)
  PartitionedVectors.set!(py, y)
  @test NLPModels.cons(nlp, y) ≈ Vector(NLPModels.cons(psnlp, py))
  @test Matrix(NLPModels.jac(nlp, y)) ≈ Matrix(NLPModels.jac(psnlp, py))
end

@testset "constraints (unconstrained models still report ncon = 0)" begin
  n = 10
  nlp = ADNLPProblems.arwhead(; n)
  psnlp = PSNLPModel(nlp)
  @test psnlp.meta.ncon == 0
  @test isempty(psnlp.vec_cons)
  @test NLPModels.cons(psnlp, psnlp.meta.x0) == []
end
