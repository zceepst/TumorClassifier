module LeNet

using ..TumorClassifier
using Flux
using Flux: onehotbatch, onecold, logitcrossentropy
using Statistics, Random
using CUDA, BSON
using ProgressMeter: @showprogress
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using Logging: with_logger

CUDA.allowscalar(false) # disable scalar GPU indexing

"Model and training parameters data type"
Base.@kwdef mutable struct Args
    η = 3e-4            # learning rate
    λ = 0               # L2 regularizer term (weight decay)
    batchSize = 128     # batch size
    ε = 10              # training epochs
    seed = 0            # seed > 0 for reproducibility (Random.seed)
    useCuda = true      # CUDA acceleration switch
    infoTime = 1        # report training progress every `infoTime` epochs
    checkTime = 5       # save model during training every `checkTime` epochs
    tbLogger = true     # TensorBoard logging switch
    savePath = "runs/"  # model results path
end

"Model constructor"
function leNet(; imgSize=(512,512,1), nClasses=4,
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
    xTrain, Ytrain = TumorClassifier.Data.loadData("../data/label.csv", "../data/image/png/";
                                                   labelClass=args.classifierType, dataType=Float32)

end

end # module
