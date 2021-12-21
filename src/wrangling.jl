module Data

using CSV, DataFrames
using Printf, Images

const fwrdLabelMap = Dict(
    "no_tumor"          => 1,
    "glioma_tumor"      => 2,
    "pituitary_tumor"   => 3,
    "meningioma_tumor"  => 4
)
const backLabelMap = Dict(
    1 => "no_tumor",
    2 => "glioma_tumor",
    3 => "pituitary_tumor",
    4 => "meningioma_tumor"
)

label2Float(x::String, map::Dict) = map[x]
float2Label(x::String, map::Dict) = map[x]
sprintfPath(dirPath, i) = dirPath * @sprintf("IMAGE_%04i.png", i)

"Encode string output labels into Float labels"
function encodeLabels(rawLabels::Vector{String};
                      labelType=:multi, type=Float32)
    if labelType == :multi
        return map(x -> label2Float(x, fwrdLabelMap), rawLabels)
    elseif labelType == :binary
        return map(x -> x == "no_tumor" ? type(0) : type(1), rawLabels)
    elseif labelType == :multiovr
        multiLabels = map(x -> label2Float(x, fwrdLabelMap), rawLabels)
        return oneVsRestLabels(multiLabels)
    end
end

"Load training and test MRI image data as: X = Array{T::dataType, 3} and Y = Array{T::dataType, 1}"
function loadData(labelPath::String, imgPath::String;
                  labelClass=:multi, dataType=Float32)::Tuple
    d = CSV.File(labelPath) |>
        DataFrame .|>
        String
    Y = encodeLabels(d.label; labelType=labelClass)
    vecX = [Images.load(sprintfPath(imgPath, i)) .|> dataType for i in 0:2999]
    X = reshapeInput(vecX, dataType)
    if labelClass == :multiovr
        X = reshapeOVR(X)
        @info "Reshaping input features, unpacking image matrices to one-dimensional vectors"
    end
    return (X, Y)
end

"Single/Fast proc. data load and pre-processing"
function loadData(dirtyX, dirtyY; labelClass=:multi, dataType=Float32)
    Y = encodeLabels(dirtyY.label; labelType=labelClass, type=dataType)
    X = reshapeInput(dirtyX, dataType)
    if labelClass == :multiovr
        X = reshapeOVR(X, dataType)
    end
    return(X, Y)
end

"Single/Fast read proc. label loading"
function readLabels(labelPath::String)
    Y = CSV.File(labelPath) |> DataFrame .|> String
    return Y
end

"Single/Fast read proc. image loading"
function readImages(imgPath::String, setSize=2999; dataType=Float32)
    return [Images.load(sprintfPath(imgPath, i)) .|> dataType for i in 0:setSize]
end

"""
Reshape a vector of matrices: Array{Array{T, 2}, 1} into Array{T, 3}.
For use in CNN classifier.

# Example:
``` julia
size(X) = (3000,)
size(reshapeInput(X)) = (512, 512, 3000)
```
"""
function reshapeInput(X::Vector{Matrix{T}}, dataType=Float32, dims=512) where {T}
    Xr = Array{dataType}(undef, dims, dims, length(X))
    for i in 1:length(X)
        Xr[:, :, i] = X[i]
    end
    return Xr
end

"""
    reshapeOVR(X::Array{T, 3}, dataType=Floa32) where {T}

One-vs-Rest input X reshaping for SVM models.
Reshapes 3D array to 1D array of Vectors (each is 'unzipped' image matrix; image -> vector).

# Example:
``` julia
size(X) = (512, 512, 3000)
size(reshapeOVR(X)) = (512^2, 3000)
```
"""
function reshapeOVR(X::Array{T, 3}, dataType=Float32) where {T}
    hX = length(X[:,1,1])
    wX = length(X[1,:,1])
    lX = length(X[1,1,:])
    Xr = Array{dataType}(undef, wX*hX, lX)
    for i in 1:lX
        Xr[:, i] = reshape(X[:,:,i], (1, wX*hX))
    end
    return Xr
end

"Split X and Y into training and testing batches"
# test: removed `args` from function parameters and `Random.seed!(args.seed)` before randperm. 
function splitTrainTest(X::Array{T1, 3}, Y::Array{T2, 1};
                        isRandom=false, split=6) where {T1, T2}
    if isRandom
        #Random.seed!(args.seed)
        testIdx = randperm(length(Y))[1:div(length(Y), split)] # random 1/split test set
        trainIdx = setdiff(1:length(Y), testIdx) # all indices of X & Y not in textIdx
        return (
            (
                selectdim(X, 3, trainIdx),  # train X
                Y[trinIdx]                  # train Y
            ),
            (
                selectdim(X, 3, testIdx),   # test X
                Y[testIdx]                  # test Y
            )
        )
    else
        testLength = div(length(Y), split)
        endTrainIdx = length(Y) - testLength
        return (
            (
                selectdim(X, 3, 1:endTrainIdx), # train X
                Y[1:endTrainIdx]                # train Y
            ),
            (
                selectdim(X, 3, endTrainIdx+1:length(Y)),   # test X
                Y[endTrainIdx+1:length(Y)]                  # test Y
            )
        )
    end
end

"Split Y labels (multiclass) into arrays corresponding to each label BitVector"
function oneVsRestLabels(encodedY)
    return [encodedY .== i for i in sort(unique(encodedY))]
end

end # module
