#=
Data wrangling for tumor detection dataset.

Dependencies:
`using CSV, DataFrames, Printf`
=#

raw = CSV.File("data/label.csv") |> DataFrame

# binary case:
# generate binary labels for brain tumor images == 'Y_bin'
raw.label = map(x -> x == "no_tumor" ? false : true,
                raw.label)


#= Import images data input == 'X' =#

# using @sprintf macro to format zeroes-prefixed integer
imagePaths = ["data/image/IMAGE_" * @sprintf("%04i", i) * ".jpg" for i in 0:2999]
images = load.(img_paths)

# general case:
