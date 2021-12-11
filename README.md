# TumorClassifier

ELEC0134 project at UCL for classifying brain scans and their corresponding tumor diagnoses.

## Coverage:

Development and runtime on JuliaHub cloud instance provider.
More information available [here](https://juliahub.com/lp/.

Instance details:

- Julia v1.6 binaries
- Intel XEON CPU E5-2686 v4 @ 2.30GHz
	- 8 vCPU
	- 16 GB memory / vCPU
- NVIDIA Tesla V100-SXM2 16GB
	- CUDA version 11.2
	- Driver version 460.73.01

## Contents:

1. Main program: `src/TumorClassifier.jl`
2. Data pipelines and wrangling: `src/wrangling.jl`
   - exports module: `Data`
3. Classifier: `src/`
   1. Support vector machine: `src/multi/svm.jl`
      - exports module: `SVM`
   2. Convolutional neural network: `src/multi/cnn.jl`
      - exports module: `LeNet5`
4. Visualization: `src/viz.jl`
5. Unit testing: `test/`

## Dependencies:

See `Project.toml` and `Manifest.toml` for a detailed breakdown of libraries used and their recursive dependencies.

```
[587475ba] Flux v0.12.8
[916415d5] Images v0.25.0
[336ed68f] CSV v0.9.11
[de0858da] Printf
[a93c6f00] DataFrames v1.3.0
[fbb218c0] BSON v0.3.4
[6e4b80f9] BenchmarkTools v1.2.1
[052768ef] CUDA v3.5.0
[56ddb016] Logging
[10745b16] Statistics
[d96e819e] Parameters v0.12.3
[92933f4c] ProgressMeter v1.7.1
[899adc3e] TensorBoardLogger v0.1.18
```

## Data Set:

> Brain Tumor Classification (MRI)

Available at [this](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri) link.
