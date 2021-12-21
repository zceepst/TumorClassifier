module CNN

using ..TumorClassifier
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Statistics, Random
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
using Images
using CUDA
import MLDatasets
import BSON

"LeNet5 constructor"
function LeNet5(; imgsize=(28,28,1), nclasses=10)
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    return Chain(
        Conv((5, 5), imgsize[end]=>6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6=>16, relu),
        MaxPool((2, 2)),
        flatten,
        Dense(prod(out_conv_size), 120, relu),
        Dense(120, 84, relu),
        Dense(84, nclasses)
    )
end

"""
    binaryGetData(args)

Function to assemble training and testing data `DataLoader` data.
Returns tuple of training and testing data loaders.
"""
function getData(args)
    XTrain, yTrain, XTest, yTest = trainData(args.labelType)
    labelEnc = args.labelType == :binary ? (0:1) : (1:4) # setting label encoding based on task A or B
    yTrain, yTest = onehotbatch(yTrain, labelEnc), onehotbatch(yTest, labelEnc)
    trainLoader = DataLoader((XTrain, yTrain), batchsize=args.batchsize, shuffle=true)
    testLoader = DataLoader((XTest, yTest), batchsize=args.batchsize)
    return trainLoader, testLoader
end

"""
    trainData(class=:binary)

Labelclass aware data fetching utlity.
Returns pre-encode/load X, y splits.
"""
function trainData(class=:binary)
    # look to make dims = 90 part of Args later
    X = TumorClassifier.locTrain.images |>
        x -> imresize.(x, 90, 90) |>
        x -> TumorClassifier.Data.reshapeInput(x, Float32, 90)
    # reshape(x, 90, 90, 1, :) after split, avoid D-loss 
    perm = randperm(3000)
    # train = 2500, test = 500 samples
    XTrain = X[:,:,perm[1:2500]] |> 
        x -> reshape(x, 90, 90, 1, :)
    XTest = X[:,:,perm[2501:3000]] |>
        x -> reshape(x, 90, 90, 1, :)
    y = TumorClassifier.locTrain.labels.label |>
        x -> TumorClassifier.Data.encodeLabels(x; labelType=class, type=Int64)
    yTrain = y[perm[1:2500]]
    yTest = y[perm[2501:3000]]
    return (XTrain, yTrain, XTest, yTest)
end

"Data loader for evaluating the model on the test set (200 images)"
function testDataLoader(batchsize, class=:binary)
    X = TumorClassifier.locTest.images |>
        x -> imresize.(x, 90, 90) |>
        x -> TumorClassifier.Data.reshapeInput(x, Float32, 90) |>
        x -> reshape(x, 90, 90, 1, :)
    y = TumorClassifier.locTest.labels.label |>
        x -> TumorClassifier.Data.encodeLabels(x; labelType=class, type=Int64)
    labelEnc = class == :binary ? (0:1) : (1:4)
    y = onehotbatch(y, labelEnc)
    return DataLoader((X, y), batchsize=batchsize)
end

"Learning loss function"
loss(ŷ, y) = logitcrossentropy(ŷ, y)

"Accuracy and loss evaluator on test points X, y"
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
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

## utility functions
"Returns the number of parameters in an NNlib model"
num_params(model) = sum(length, Flux.params(model)) 
"Round to 4 digits"
round4(x) = round(x, digits=4)

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 128      # batch size
    epochs = 10          # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    use_cuda = true      # if true use cuda (if available)
    infotime = 1 	     # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true      # log training with tensorboard
    savepath = "runs/"      # results path
    modelName = "LeNet5"    # .bson save name
    labelType = :binary     # classifer 'task' selector (bin/multi)
end

"""
    trainClass(; kwargs...)

Main training routine for CNN model.
Modify the training characteristics by passing keyword-args to function call.

See `TumorClassifier.CNN.Args` for keyword possibilities.
"""
function trainClass(; kwargs...)
    args = Args(; kwargs...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()

    if use_cuda
        device = gpu
        @info "Training on GPU!"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## DATA
    train_loader, test_loader = getData(args)
    @info "Dataset MRI scans: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    numClasses = args.labelType == :binary ? 2 : 4
    model = LeNet5(; imgsize=(90,90,1), nclasses=numClasses) |> device
    @info "LeNet5 model: $(num_params(model)) trainable params"
    
    ps = Flux.params(model)

    opt = ADAM(args.η)
    if args.λ > 0 # add weight decay, equivalent to L2 regularization
        opt = Optimiser(WeightDecay(args.λ), opt)
    end
    
    ## LOGGING UTILITIES
    if args.tblogger
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end
    
    function report(epoch)
        train = evalLossAccuracy(train_loader, model, device)
        test = evalLossAccuracy(test_loader, model, device)
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss=train.loss  acc=train.acc
                @info "test"  loss=test.loss   acc=test.acc
            end
        end
    end
    
    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        @showprogress for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do # backprop grad over hyp-params
                    ŷ = model(x) # make current model pred. on current model
                    loss(ŷ, y)
                end

            Flux.Optimise.update!(opt, ps, gs) # where the hyp-params get upd.
        end
        
        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, args.modelName * ".bson") 
            let model = cpu(model) #return model to cpu before serialization
                BSON.@save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end

"evaluate final model performace on released test set (200 images dataset)"
function evalModelWithTestSet(modelPath, labelType, batchsize, useCuda=false)
    model = BSON.load(modelPath)[:model]
    useCuda = useCuda && CUDA.functional()    
    if useCuda
        device = gpu
        @info "Inference on GPU!"
    else
        device = cpu
        @info "Inference on CPU"
    end
    loader = testDataLoader(batchsize, labelType)
    return evalLossAccuracy(loader, model, device)
end

end # module