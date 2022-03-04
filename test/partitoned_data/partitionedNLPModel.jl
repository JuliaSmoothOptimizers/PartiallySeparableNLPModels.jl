using OptimizationProblems
using NLPModels, NLPModelsJuMP

@testset "test PBGSNLPModel et PLBFGSNLPModel" begin
	n = 10
	nlp = MathOptNLPModel(OptimizationProblems.arwhead(n), name="arwhead " * string(n))

	pbfgsnlp = PBFGSNLPModel(nlp)
	pblfgsnlp = PBFGSNLPModel(nlp)

	@test NLPModels.obj(nlp) == NLPModels.obj(pbfgsnlp)
	@test NLPModels.obj(nlp) == NLPModels.obj(plbfgsnlp)
end 