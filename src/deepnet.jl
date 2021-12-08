module LeNet

using Flux
using Flux: onehotbatch, onecold, logitcrossentropy
using Statistics, Random
using CUDA, BSON
using ProgressMeter: @showprogress
using TensorBoardLoagger: TBLogger, tb_overwrite, set_step!, set_step_incremement!
using Logging: with_logger

CUDA.allowscalar(false) # disable scalar GPU indexing

"Model constructor"



end # module
