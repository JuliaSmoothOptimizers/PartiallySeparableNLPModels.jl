
@testset "test PBGSNLPModel et PLBFGSNLPModel MathOptNLPModel" begin
  n = 10
  nlp = MathOptNLPModel(OptimizationProblems.arwhead(n), name = "arwhead " * string(n))
  x = rand(n)

  pbfgsnlp = PBFGSNLPModel(nlp)
  plbfgsnlp = PLBFGSNLPModel(nlp)

  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(pbfgsnlp, x)
  @test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plbfgsnlp, x)

  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(pbfgsnlp, x)
  @test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plbfgsnlp, x)
end

# @testset "test PBGSNLPModel et PLBFGSNLPModel ADNLPModel" begin
# 	start_ones(n :: Int) = ones(n)
# 	function arwhead(x :: AbstractVector{Y}) where Y <: Number
# 		n = length(x)
# 		n < 2 && @warn("arwhead: number of variables must be ≥ 2")
# 		n = max(2, n)

# 		return sum((x[i]^2 + x[n]^2)^2 - 4 * x[i] + 3 for i=1:n-1)
# 	end
# 	start_arwhead(n :: Int) = ones(n)
# 	arwhead_ADNLPModel(n :: Int=100) = ADNLPModel(arwhead, start_arwhead(n), name="arwhead "*string(n) * " variables")

# 	n = 10
# 	x = rand(n)
# 	nlp = arwhead_ADNLPModel(n)

# 	pbfgsnlp = PBFGSNLPModel(nlp)
# 	plbfgsnlp = PLBFGSNLPModel(nlp)

# 	@test NLPModels.obj(nlp, x) ≈ NLPModels.obj(pbfgsnlp, x)
# 	@test NLPModels.obj(nlp, x) ≈ NLPModels.obj(plbfgsnlp, x)

# 	@test NLPModels.grad(nlp, x) ≈ NLPModels.grad(pbfgsnlp, x)
# 	@test NLPModels.grad(nlp, x) ≈ NLPModels.grad(plbfgsnlp, x)
# end 

# using Test
# using CalculusTreeTools, PartiallySeparableNLPModels

# using JuMP, MathOptInterface, LinearAlgebra, SparseArrays
# using ADNLPModels, NLPModels, NLPModelsJuMP
# using OptimizationProblems

# start_ones(n :: Int) = ones(n)
# function arwhead(x :: AbstractVector{Y}) where Y <: Number
# 	n = length(x)
# 	n < 2 && @warn("arwhead: number of variables must be ≥ 2")
# 	n = max(2, n)

# 	return sum((x[i]^2 + x[n]^2)^2 - 4 * x[i] + 3 for i=1:n-1)
# end
# start_arwhead(n :: Int) = ones(n)
# arwhead_ADNLPModel(n :: Int=100) = ADNLPModel(arwhead, start_arwhead(n), name="arwhead "*string(n) * " variables")

# n = 10
# x = rand(n)

# nlp_ad = arwhead_ADNLPModel(n)
# nlp_maopt = MathOptNLPModel(OptimizationProblems.arwhead(n), name="arwhead " * string(n))

# pbfgsnlp_ad = PBFGSNLPModel(nlp_ad)
# plbfgsnlp_ad = PLBFGSNLPModel(nlp_ad)

# pbfgsnlp_maopt = PBFGSNLPModel(nlp_maopt)
# plbfgsnlp_maopt = PLBFGSNLPModel(nlp_maopt)

# NLPModels.obj(nlp_ad,x)	
# NLPModels.obj(nlp_maopt,x)

# NLPModels.obj(pbfgsnlp_ad,x)
# NLPModels.obj(plbfgsnlp_ad,x)

# NLPModels.obj(pbfgsnlp_maopt,x)
# NLPModels.obj(plbfgsnlp_maopt,x)
