using Test

using LinearAlgebra
using ADNLPModels, NLPModels, NLPModelsJuMP
using OptimizationProblems, OptimizationProblems.ADNLPProblems, OptimizationProblems.PureJuMP
using JSOSolvers
using PartiallySeparableNLPModels

include("pqnnlp.jl")
include("methods.jl")
