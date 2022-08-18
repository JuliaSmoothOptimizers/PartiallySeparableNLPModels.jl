using Documenter
using PartiallySeparableNLPModels
using PartiallySeparableNLPModels: ModAbstractPSNLPModels, Utils, ModPBFGSNLPModels, ModPCSNLPModels, ModPLBFGSNLPModels, ModPLSENLPModels, ModPLSR1NLPModels, ModPSENLPModels, ModPSNLPModels, ModPSR1NLPModels

makedocs(
  modules = [
    PartiallySeparableNLPModels,
    ModAbstractPSNLPModels,
    Utils,
    ModPBFGSNLPModels,
    ModPCSNLPModels,
    ModPLBFGSNLPModels,
    ModPLSENLPModels,
    ModPLSR1NLPModels,
    ModPSENLPModels,
    ModPSNLPModels,
    ModPSR1NLPModels
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
