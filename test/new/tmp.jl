
adnlp = ADNLPProblems.rosenbrock(; n)
pqnnlp = PVQNPModel(adnlp)

@test NLPModels.obj(adnlp, adnlp.meta.x0) == NLPModels.obj(pqnnlp, pqnnlp.meta.x0)

@test Vector(NLPModels.grad(pqnnlp, pqnnlp.meta.x0)) == NLPModels.grad(adnlp, adnlp.meta.x0)