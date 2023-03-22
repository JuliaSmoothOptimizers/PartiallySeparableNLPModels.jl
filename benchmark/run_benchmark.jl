using PkgBenchmark

commit = benchmarkpkg("PartiallySeparableNLPModels")  #dernier commit sur la branche sur laquelle on se trouve
master = benchmarkpkg("PartiallySeparableNLPModels", "master") # branche master
judgement = judge(commit, master)

export_markdown("benchmark/judgement.md", judgement)
