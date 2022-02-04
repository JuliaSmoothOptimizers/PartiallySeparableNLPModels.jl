using Documenter
using PartiallySeparableNLPModels

makedocs(
  modules = [PartiallySeparableNLPModels],
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

deploydocs(repo = "github.com/paraynaud/PartiallySeparableNLPModels.jl.git", devbranch = "master")