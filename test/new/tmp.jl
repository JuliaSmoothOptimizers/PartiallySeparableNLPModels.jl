
adnlp = ADNLPProblems.rosenbrock(; n)
pqnnlp = PVQNPModel(adnlp)

@test NLPModels.obj(adnlp, adnlp.meta.x0) == NLPModels.obj(pqnnlp, pqnnlp.meta.x0)

@test Vector(NLPModels.grad(pqnnlp, pqnnlp.meta.x0)) == NLPModels.grad(adnlp, adnlp.meta.x0)

NLPModels.hprod!(pqnnlp, pqnnlp.meta.x0, pv, Hv)
NLPModels.hprod!(adnlp, adnlp.meta.x0, v, hv)

v = ones(n)
pv = similar(pqnnlp.meta.x0; simulate_vector=true)
set!(pv, v)
Hv = similar(pqnnlp.meta.x0; simulate_vector=false)
Vector(Hv) == hv

#not true because pqnnlp is a PQN NLPmodel
# Vector(NLPModels.hprod(pqnnlp, pqnnlp.meta.x0, pv)) == NLPModels.hprod(adnlp, adnlp.meta.x0, v)

NLPModels.hprod!(pqnnlp, pqnnlp.meta.x0, pv, Hv)
hv = Matrix(pqnnlp.pB) * v
Vector(Hv) == hv

Hv2 = similar(Hv)
B = hess_op!(pqnnlp, pqnnlp.meta.x0, Hv2)
Hv2 == Hv
Vector(Hv2) == hv