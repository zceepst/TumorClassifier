module LeNet

using ..TumorClassifier
using Flux
using Flux: onehotbatch, onecold
using Flux.Optimise: Optimiser, WeightDecay
using Flux.Losses: logitcrossentropy
using Flux: DataLoader
using Statistics, Random
using CUDA, BSON
using ProgressMeter
using ProgressMeter: @showprogress
using TensorBoardLogger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using Logging
using Logging: with_logger

CUDA.allowscalar(false) # disable scalar GPU indexing

"Model and training parameters data type"
Base.@kwdef mutable struct Args
    classifierType = :multi # classifier problem switch (:multi/:binary)
    η = 3e-4            # learning rate
    λ = 0               # L2 regularizer term (weight decay)
    batchSize = 128     # batch size
    epochs = 10         # training epochs
    seed = 2            # seed > 0 for reproducibility (Random.seed)
    useCuda = true      # CUDA acceleration switch
    infoTime = 1        # report training progress every `infoTime` epochs
    checkTime = 5       # save model during training every `checkTime` epochs
    tbLogger = true     # TensorBoard logging switch
    savePath = "runs/"  # model results path
end

"Model constructor"
function leNetModel(; imgSize=(512,512,1), nClasses=4,
               outputConvSize=(div(imgSize[1], 4) - 3, div(imgSize[2], 4) - 3, 16))
    return Chain(
        Conv( (5,5), imgSize[end] => 6, relu ),
        MaxPool( (2,2) ),
        flatten,
        Dense( prod(outputConvSize), 120, relu ),
        Dense( 120, 84, relu ),
        Dense( 84, nClasses )
    )
end

"Prepare training and testing data sources, then load into DataLoader type for supply to model train/test routine, return trainLoader, testLoader tuple"
function dataPrepGet(args)
    X, Y = TumorClassifier.Data.loadData("data/label.csv", "data/image/png/";
                                         labelClass=args.classifierType, dataType=Float32)
    (xTrain, yTrain), (xTest, yTest) = TumorClassifier.Data.splitTrainTest(X, Y; isRandom=false, split=2)
    # reshape train/test inputs to allow for convolution layers at later stages

    @info "Size X train: $(size(xTrain))    Size X test: $(size(xTest))"
    @info "Size Y train: $(size(yTrain))    Size Y test: $(size(yTest)))"

    xTrain = reshape(xTrain, 512, 512, 1, :)
    xTest = reshape(xTest, 512, 512, 1, :)

    @info "Post reshape:"
    @info "Size X train: $(size(xTrain))    Size X test: $(size(xTest))"

    # encode with Flux.onehotbatch
    yTrain, yTest = Flux.onehotbatch(yTrain, 1:4), Flux.onehotbatch(yTest, 1:4)
    @info "Post encoded"
    @info "Size Y train: $(size(yTrain))    Size Y test: $(size(yTest))"

    # wrap data with Flux.DataLoader
    trainLoader = Flux.DataLoader(
        (xTrain, yTrain),
        batchsize = args.batchSize,
        shuffle = true
    )
    testLoader = Flux.DataLoader(
        (xTrain, yTest),
        batchsize = args.batchSize
    )
    return (trainLoader, testLoader)
end

"Custom loss function"
loss(ŷ, y) = Flux.logitcrossentropy(ŷ, y)
"Shortcut rounding function"
round4(x) = round(x, digits=4)


"Loss evaluation function, return the loss and accuracy of the model on data X, Y as a named tuple: (loss = ..., acc = ...)"
function evalLossAccuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (
        loss = l/ntot |> round4,
        acc = acc/ntot*100 |> round4
    )
end

"Evaluate the number of parameters of a Flux model"
numParams(model) = sum(length, Flux.params(model))

"""
    train(; kwargs...)

LeNet training/evaluation function.
Reads, encodes, splits into train and test, and wraps X, Y with the `DataLoader` type to feed to model.

Pass keywords arguments (`kwargs`) to `Args` type to modify:

- training epochs: `Args.epoch`
- learning reate: `Args.η`
- weight decay: `Arg.λ`
- batch size: `Args.batchSize`
- RNG seed: `Args.seed`
- GPU device switch: `Args.useCUDA`
- logging behaviour: `Args.checkTime`, `Args.infoTime`, `Args.tbLogger`
- parameter, model and logging saving path: `Args.savePath`
"""
function train(; kwargs...)
    args = Args(; kwargs...)                    # splat keyword-args from function arguments
    args.seed > 0 && Random.seed!(args.seed)    # if seed provided, generate reproducible rng
    useCUDA = args.useCuda && CUDA.functional() # true if GPU/CUDA device available

    # set training device and log info
    if useCUDA
        device = gpu
        @info "Training on GPU!"
    else
        device = cpu
        @info "Training on CPU."
    end

    ## Load data:
    trainLoader, testLoader = dataPrepGet(args)
    @info "Dataset: MRI brain scans: $(trainLoader.nobs) train and $(testLoader.nobs) test examples."

    ## Model and optimiser:
    model = leNetModel() |> device # pipe to cpu/gpu
    @info "LeNet model: $(numParams(model)) trainable parameters."

    ps = Flux.params(model)

    opt = ADAM(args.η) # ADAM optimiser
    if args.λ > 0 # weight decay / L2 reg.
        opt = Flux.Optimiser(Flux.WeightDecay(args.λ), opt)
    end

    ## Logging utils:
    if args.tbLogger
        tbLogger = TensorBoardLogger.TBLogger(args.savePath, TensorBoardLogger.tb_overwrite)
        TensorBoardLogger.set_step_increment!(tbLogger, 0) # manual set_step! so set 0 local
        @info "TensorBoard logging at \"$(args.savePath)\"."
    end

    "Training progress report logging function"
    function report(epoch)
        train = evalLossAccuracy(trainLoader, model, device)
        test = evalLossAccuracy(testLoader, model, device)
        println("Epoch: $epoch      Train: $(train)     Test: $(test)")
        if args.tbLogger
            TensorBoardLogger.set_step!(tbLogger, epoch)
            Logging.with_logger(tbLogger) do
                @info "train"   loss=train.loss     acc=train.acc
                @info "test"    loss=test.loss      acc=test.acc
            end
        end
    end

    ## Training:
    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        ProgressMeter.@showprogress for (x, y) in trainLoader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        ## Printing and logging training:
        mod(epoch, args.infoTime) == 0 && report(epoch)
        if args.checkTime > 0 && mod(epoch, args.checkTime) == 0
            !ispath(args.savePath) && mkpath(args.savePath) # create logging path if not exist
            modelPath = joinpath(args.savePath, "model.bson")
            let model = cpu(model) # revert model back to cpu for save
                BSON.@save modelPath model epoch # BSON save macro
            end
            @info "Model saved in \"$(modelPath)\""
        end
    end
end

end # module
