#= Tumor Classifier main program =#

using Flux

include("wrangling.jl") # provides Data module with pipelines and cleaning

binaryScanData() = Data.sourceScans("data/label.csv", "data/image/png/", labelClass=:binary)
multiScanData() = Data.sourceScans("data/label.csv", "data/image/png/", labelClass=:multi)
