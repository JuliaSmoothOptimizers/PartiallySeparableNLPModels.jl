module PartiallySeparableNLPModels

using ExpressionTreeForge
using PartitionedStructures

include("AbstractPNLPModels.jl")
include("partitionedNLPModels/_include.jl")
include("trunk.jl")

using .ModAbstractPSNLPModels
using .ModPBFGSNLPModels, .ModPLBFGSNLPModels, .ModPCSNLPModels, .ModPLSR1NLPModels, .ModPLSENLPModels, .ModPSR1NLPModels, .ModPSENLPModels, .ModPSNLPModels
using .TrunkInterface

export PartiallySeparableNLPModel, AbstractPQNNLPModel
export element_function
export PBFGSNLPModel, PCSNLPModel, PLBFGSNLPModel, PLSR1NLPModel, PLSENLPModel, PSR1NLPModel, PSENLPModel, PSNLPModel

# export product_part_data_x, evaluate_obj_part_data, evaluate_grad_part_data
# export product_part_data_x!,
#   evaluate_obj_part_data!, evaluate_y_part_data!, evaluate_grad_part_data!

end
