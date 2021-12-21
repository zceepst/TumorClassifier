#= Tumor classifer main program module =#
module TumorClassifier

using LIBSVM

include("wrangling.jl") # provides `Data` module with pipelines and cleaning
include("svc.jl")
include("cnn.jl") # provides `LeNet` module for CNN classifier

# database constant directory paths
const PNG_PATH = "data/image/png/"
const LABEL_PATH = "data/label.csv"
const TEST_PNG_PATH = "data/test/image/png/"
const TEST_LABEL_PATH = "data/test/label.csv"

struct Memory
    labels
    images
end

function localMemory(pathImages, pathLabels, setSize=2999)
    return Memory(
        TumorClassifier.Data.readLabels(pathLabels),
        TumorClassifier.Data.readImages(pathImages, setSize; dataType=Float32)
    )
end

# train- and testing local memory image data i/o
# read in at module load to save time later; only calls n x Images.load() twice
locTrain = localMemory(PNG_PATH, LABEL_PATH)
locTest = localMemory(TEST_PNG_PATH, TEST_LABEL_PATH, 199)

binarySVC() = TumorClassifier.SVC.trainEval(NuSVC, true)

end #module
