module Data

using CSV, DataFrames
using Printf, Images

const fwrdLabelMap = Dict(
    "no_tumor"          => 1.0,
    "glioma_tumor"      => 2.0,
    "pituitary_tumor"   => 3.0,
    "meningioma_tumor"  => 4.0
)
const backLabelMap = Dict(
    1.0 => "no_tumor",
    2.0 => "glioma_tumor",
    3.0 => "pituitary_tumor",
    4.0 => "meningioma_tumor"
)

label2Float(x::String, map::Dict) = map[x]
float2Label(x::String, map::Dict) = map[x]
sprintfPath(dirPath, i) = dirPath * @sprintf("IMAGE_%04i.png", i)

function encodeLabels(rawLabels::Vector{String};
                      labelType=:multi)::Vector{Float64}
    if labelType == :multi
        return map(x -> label2Float(x, fwrdLabelMap), rawLabels)
    elseif labelType == :binary
        return map(x -> x == "no_tumor" ? 0.0 : 1.0, rawLabels)
    end
end

function loadData(labelPath::String, imgPath::String;
                  labelClass=:multi, dataType=Float32)::Tuple
    d = CSV.File(labelPath) |>
        DataFrame .|>
        String
    Y = encodeLabels(d.label; labelType=labelClass) .|> dataType
    X = [Images.load(sprintfPath(imgPath, i)) .|> dataType for i in 0:2999]
    return (X, Y)
end

end # module
