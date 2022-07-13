using Documenter
using PartiallySeparableNLPModels
using PartiallySeparableNLPModels:
  Mod_ab_partitioned_data, Mod_PQN, Mod_common, Mod_partitionedNLPModel

makedocs(
  modules = [
    PartiallySeparableNLPModels,
    Mod_ab_partitioned_data,
    Mod_common,
    Mod_partitionedNLPModel,
    Mod_PQN,
  ],
  doctest = true,
  # linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "PartiallySeparableNLPModels.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/PartiallySeparableNLPModels.jl.git",
  push_preview = true,
  devbranch = "master",
)
