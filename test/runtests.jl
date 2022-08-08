using Test
using PartiallySeparableNLPModels

using ExpressionTreeForge, PartitionedStructures
using JuMP, MathOptInterface, LinearAlgebra
using ADNLPModels, NLPModels, NLPModelsJuMP
using OptimizationProblems, OptimizationProblems.ADNLPProblems, OptimizationProblems.PureJuMP

include("partitoned_data/_include.jl")
