using Documenter
using PartiallySeparableNLPModel

makedocs(
  modules = [PartiallySeparableNLPModel],
  doctest = true,
  # linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "PartiallySeparableNLPModel.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(repo = "github.com/paraynaud/PartiallySeparableNLPModel.jl.git", devbranch = "main")
