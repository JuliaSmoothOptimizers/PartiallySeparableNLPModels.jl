using PkgBenchmark
using SolverBenchmark
# import PartiallySeparableStructure


commit = benchmarkpkg("PartiallySeparableNLPModels")  #dernier commit sur la branche sur laquelle on se trouve
master = benchmarkpkg("PartiallySeparableNLPModels", "master") # branche master
judgement = judge(master, commit)
# judgement = judge("PartiallySeparableStructure", "master")
#commentaire initile
export_markdown("benchmark/judgement.md", judgement)
