#= Tumor Classifier main program =#
module TumorClassifier

using Flux

include("wrangling.jl") # provides `Data` module with pipelines and cleaning
include("cnn.jl") # provedes `LeNet` module for training and evaluation of CNN model

binaryScanData() = Data.sourceScans("data/label.csv", "data/image/png/", labelClass=:binary)
multiScanData() = Data.sourceScans("data/label.csv", "data/image/png/", labelClass=:multi)


end #module
