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
function LeNet(; imgSize=(512,512,1), nClasses=4
               outputConvSize=(div(imageSize[1], 4) - 3, div(imageSize[2], 4) - 3, 16))
    return Chain(
        Conv( (5,5), imgSize[end] => 6, relu ),
        MaxPool( (2,2) ),
        Conv( (5,5), 6 => 16, relu ),
        MaxPool( (2,2) ),
        flatten,
        Dense( prod(outputConvSize), 120, relu ),
        Dense( 120, 84, relu ),
        Dense( 84, nClasses )
    )
end

"Reduce Float64"
function dataPrep(args)
    xTrain, Ytrain =
end

end # module
