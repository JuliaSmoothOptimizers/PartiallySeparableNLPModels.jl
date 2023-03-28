using BenchmarkTools
using NLPModels, ADNLPModels, OptimizationProblems
using PartiallySeparableNLPModels

const SUITE = BenchmarkGroup()

# element functions of two variables
small_elements(x; n = length(x)) = sum(x[1] * x[i] for i = 1:n) + sum(x[n] * x[i] for i = 1:n)
x0_small_elements(n::Int) = ones(n)
small_elements_model(n::Int) = ADNLPModel(small_elements, x0_small_elements(n))

# element function whose size grows as n grows
function large_elements(x; n = length(x), div = Int(floor(sqrt(n))))
  f = Int(floor(n / div))
  sub(range) = sum(i * x[i] for i in range)^2
  sum(sub(((i - 1) * f + 1):((i + 2) * f)) / (x[i]^2) for i = 1:(div - 3)) +
  sum(sub(((i - 1) * f + 5):((i + 4) * f + 5)) / (x[n - i]^2) for i = 1:(div - 5)) +
  (x[1] * x[n]^3)^2
end
function x0_large_elements(n::Int; div = Int(floor(sqrt(n))))
  f = Int(floor(n / div))
  _n = f * div
  10 .* ones(_n)
end
large_elements_model(n::Int) = ADNLPModel(large_elements, x0_large_elements(n))

squared_nmin = 6
squared_nmax = 30
ns = (len -> len^2).(collect(squared_nmin:squared_nmax))

for n in ns
  SUITE["small obj $n"] = BenchmarkGroup()
  SUITE["small grad $n"] = BenchmarkGroup()
  adnlp = small_elements_model(n)
  psnlp = PSNLPModel(adnlp)

  SUITE["small obj $n"] = @benchmarkable NLPModels.obj($psnlp, $psnlp.meta.x0)
  grad = similar(psnlp.meta.x0; simulate_vector = false)
  SUITE["small grad $n"] = @benchmarkable NLPModels.grad!($psnlp, $psnlp.meta.x0, $grad)

  SUITE["large obj $n"] = BenchmarkGroup()
  SUITE["large grad $n"] = BenchmarkGroup()
  adnlp = large_elements_model(n)
  psnlp = PSNLPModel(adnlp)

  SUITE["large obj $n"] = @benchmarkable NLPModels.obj($psnlp, $psnlp.meta.x0)
  grad = similar(psnlp.meta.x0; simulate_vector = false)
  SUITE["large grad $n"] = @benchmarkable NLPModels.grad!($psnlp, $psnlp.meta.x0, $grad)
end
