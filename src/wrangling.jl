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

function encodeLabels(rawLabels::Vector{String};
                      labelType=:multi)::Vector{Float64}
    if labelType == :multi
        return map(x -> label2Float(x, fwrdLabelMap), rawLabels)
    elseif labelType == :binary
        return map(x -> x == "no_tumor" ? 0 : 1, rawLabels)
    end
end

function loadData(labelPath::String, imgPath::String;
                  labelClass=:multi, dataType=Float32)::Tuple
    d = CSV.File(labelPath) |>
        DataFrame .|>
        String
    Y = encodeLabels(d.label; labelType=labelClass)
    vecX = [Images.load(sprintfPath(imgPath, i)) .|> dataType for i in 0:2999]
    X = reshapeInput(X, dataType)
    return (X, Y)
end

function reshapeInput(X::Vector{Matrix{T}}, dataType=Float32) where {T}
    Xr = Array{dataType}(undef, 512, 512, length(X))
    for i in 1:length(X)
        Xr[:, :, i] = X[i]
    end
    return Xr
end

"Split X and Y into training and testing batches"
function splitTrainTest(X::Array{T, 3}, Y::Array{T, 1};
                        isRandom=false, split=6) where {T}
    if isRandom
        # Random.seed!(args.seed)
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
                selectdim(X, 3, 1:endTrainIdx),             # train X
                Y[1:endTrainIdx]                            # train Y
            ),
            (
                selectdim(X, 3, endTrainIdx+1:length(Y)),   # test X
                Y[endTrainIdx+1:length(Y)]                  # test Y
            )
        )
    end
end

end # module
