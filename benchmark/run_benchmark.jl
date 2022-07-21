using PkgBenchmark
using SolverBenchmark

commit = benchmarkpkg("PartiallySeparableNLPModels")  #dernier commit sur la branche sur laquelle on se trouve
master = benchmarkpkg("PartiallySeparableNLPModels", "benchmmark") # branche master
judgement = judge(master, commit)

export_markdown("benchmark/judgement.md", judgement)
