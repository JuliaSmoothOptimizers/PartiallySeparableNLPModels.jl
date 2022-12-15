module PartiallySeparableNLPModels

using ExpressionTreeForge
using PartitionedStructures

include("AbstractPNLPModels.jl")
include("partitionedNLPModels/_include.jl")
include("trunk.jl")

using .Utils, .ModAbstractPSNLPModels
using .ModPBFGSNLPModels,
  .ModPLBFGSNLPModels,
  .ModPCSNLPModels,
  .ModPLSR1NLPModels,
  .ModPLSENLPModels,
  .ModPSR1NLPModels,
  .ModPSENLPModels,
  .ModPSNLPModels
using .TrunkInterface

export PartiallySeparableNLPModel, AbstractPQNNLPModel
export element_function, partitioned_structure
export PBFGSNLPModel,
  PCSNLPModel, PLBFGSNLPModel, PLSR1NLPModel, PLSENLPModel, PSR1NLPModel, PSENLPModel, PSNLPModel

end
