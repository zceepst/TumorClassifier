struct Scan
    id::Int64
    input::Matrix{Gray{N0f8}}
    output::Float64
end

"""
    sourceScans(labelPath::String, imgPath::String; labelClass=:multi)

Brain scan data-sourcing function.
Generates a vector of type `Scan`, which individually store an: `id`, `input` which is an
image represented as a grayscale matrix and an `output` label of type `Float64`.

To modify the data according to multiclass or binary classifier use-cases, use the `labelClass`
keyword argument to specify how the label data from the `label.csv` file should be modified.

This format was orignally
"""
function sourceScans(labelPath::String, imgPath::String; labelClass=:multi)
    d = CSV.File(labelPath) |>
        DataFrame .|>
        String
    labels = encodeLabels(d.label; labelType=labelClass)
    scans = Scan[]
    # Threads.@threads for i in 0:2999
    for i in 0:2999
        push!(
            scans,
            Scan(
                i + 1,
                Images.load(sprintfPath(imgPath, i)) .|> Gray{N0f8},
                labels[i + 1]
            )
        )
    end
    @assert length(scans) == 3000
    return scans
end
