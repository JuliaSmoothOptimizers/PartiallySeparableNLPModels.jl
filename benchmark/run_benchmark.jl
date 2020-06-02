using PkgBenchmark
using SolverBenchmark
# import PartiallySeparableStructure


commit = benchmarkpkg("PartiallySeparableStructure")  #dernier commit sur la branche sur laquelle on se trouve
master = benchmarkpkg("PartiallySeparableStructure", "master") # branche master
judgement = judge(master, commit)
# judgement = judge("PartiallySeparableStructure", "master")
export_markdown("benchmark/judgement.md", judgement)
