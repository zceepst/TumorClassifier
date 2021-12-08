# TumorClassifier

ELEC0134 project at UCL for classifying brain scans and their corresponding tumor diagnoses.

## Contents

1. Main program: `src/TumorClassifier.jl`
2. Data pipelines and wrangling: `src/wrangling.jl`
3. Binary classifier: `src/binary/`
   1. SVM: `src/binary/svm.jl`
   2. CNN: `src/binary/deepnet.jl`
4. Multiclass classifier: `src/multiclass`
   1. SVM: `src/multi/svm.jl`
   2. CNN: `src/multi/deepnet.jl`
5. Visualizations: `src/viz.jl`
6. Unit testing: `test/`

## Dependencies

- `Flux.jl`
- `CSV.jl`
- `DataFrames.jl`
- `Plots.jl`
- `Printf.jl`
- `Images.jl`

## Data Set

> Brain Tumor Classification (MRI)

Available at [this](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri) link.
