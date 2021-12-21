module SVC

using ..TumorClassifier
using LIBSVM
using Images
using RDatasets
using Statistics
using LinearAlgebra
using BSON
import MLJ

# Classification C-SVM
function loadIris()
    iris = dataset("datasets", "iris")
    X = Matrix(iris[:, 1:4])
    y = iris.Species
    Xtrain = X[1:2:end, :]
    Xtest  = X[2:2:end, :]
    ytrain = y[1:2:end]
    ytest  = y[2:2:end]
    return (Xtrain, ytrain, Xtest, ytest)
end

function irisExample()
    Xtrain, ytrain, Xtest, ytest = loadIris()
    model = fit!(SVC(), Xtrain, ytrain)
    ŷ = predict(model, Xtest)
    score = mean(ŷ .== ytest) * 100
    return score
end

function rawXXyy()
    train = TumorClassifier.locTrain
    test = TumorClassifier.locTrain
    return (train.images, train.labels.label, test.images, test.labels.label)
end

function reduceImages(images, newDim)
    return imresize.(images, newDim, newDim)
end

function importData()
    Xtrain, ytrain, Xtest, ytest = rawXXyy()
    Xtrain = reduceImages(Xtrain, 90) # reduce image matrix size (compression)
    Xtest = reduceImages(Xtest, 90) # will inccur some data loss, but offer speed gains
    ytrain = MLJ.categorical(ytrain)
    ytest = MLJ.categorical(ytest)
    return (
        TumorClassifier.Data.reshapeInput(Xtrain, Float64, 90) |> x -> TumorClassifier.Data.reshapeOVR(x, Float64) |> x -> x'[:,:],
        ytrain,
        TumorClassifier.Data.reshapeInput(Xtest, Float64, 90) |> x -> TumorClassifier.Data.reshapeOVR(x, Float64) |> x -> x'[:,:],
        ytest
    )
end

function genModel(F, Xtr, ytr)
    return fit!(F(), Xtr, ytr)
end

function accuracy(ŷ, y)
    return mean(ŷ .== y) * 100
end

function binaryEncoder(ŷ, y)
    return (
        TumorClassifier.Data.encodeLabels(String.(ŷ); labelType=:binary, type=Int64),
        TumorClassifier.Data.encodeLabels(String.(y); labelType=:binary, type=Int64)
    )
end

function trainEval(Model, save=false, savePath="models/SVC/")
    @info "Loading training and testing data"
    Xtr, ytr, Xte, yte = importData()
    @info "Generating a $(Model) model, then fitting to training dataset"
    model = genModel(Model, Xtr, ytr)
    @info "Training finished"
    @info "Evaluating $(Model) performance on test set"
    ŷ = LIBSVM.predict(model, Xte)
    ŷenc, yenc = binaryEncoder(ŷ, yte)
    acc = accuracy(ŷ, yte)
    binAcc = accuracy(ŷenc, yenc)
    @info "Multiclass classifier accuracy:  $(acc)%"
    @info "Binary classifier accuracy:      $(binAcc)%"
    if save
        saveName = savePath * "$(Model)"
        @info "Saving model to BSON object file at: $(saveName)"
        BSON.@save saveName model
    end
end

end