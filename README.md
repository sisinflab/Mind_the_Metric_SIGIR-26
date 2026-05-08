# Mind the Metric: Reproducibility and Fair Benchmarking of Spectral Graph Models for Collaborative Filtering

This is the official repository for the paper "_Mind the Metric: Reproducibility and Fair Benchmarking of Spectral Graph Models for Collaborative Filtering_".

This repository provides all the necessary code, data splits, and configurations to reproduce the experiments presented in the paper. The experimental framework is built upon **Elliot**, a comprehensive and rigorous framework for reproducible recommender systems evaluation. We strongly suggest referring to the official Elliot [GitHub page](https://github.com/sisinflab/elliot) and [documentation](https://elliot.readthedocs.io/en/latest/).

## 📖 Overview

Spectral graph-based collaborative filtering (CF) models have recently gained attention for their theoretical grounding and reported state-of-the-art performance. By leveraging spectral denoising, graph signal processing (GSP), and polynomial graph filtering techniques, these approaches aim to capture high-order connectivity and frequency-aware preference signals. However, concerns remain regarding the rigor, fairness, and reproducibility of their empirical evaluation.

This work presents a systematic and fully reproducible reassessment of three major families of spectral recommenders:

| Family | Model | Paper | DOI / Link | Code Repository |
| :--- | :--- | :--- | :--- | :--- |
| **Spectral Denoising** | **SVD-GCN** | SVD-GCN: A Simplified Graph Convolution Paradigm for Recommendation | [10.1145/3511808.3557252](https://doi.org/10.1145/3511808.3557252) | [GitHub](https://github.com/tanatosuu/svd_gcn) |
| | **GDE** | Less is More: Reweighting Important Spectral Graph Features for Recommendation | [10.1145/3477495.3532046](https://doi.org/10.1145/3477495.3532046) | [GitHub](https://github.com/tanatosuu/GDE) |
| | **SGDE** / **RSGDE** / **CSGDE** | Less is More: Removing Redundancy of Graph Convolutional Networks for Recommendation | [10.1145/3632751](https://doi.org/10.1145/3632751) | [GitHub](https://github.com/tanatosuu/GDE) |
| | **SGFCF** | How Powerful is Graph Filtering for Recommendation | [10.1145/3637528.3671913](https://doi.org/10.1145/3637528.3671913) | [GitHub](https://github.com/tanatosuu/sgfcf) |
| **Graph Signal Processing (GSP)** | **PGSP** | Personalized Graph Signal Processing for Collaborative Filtering | [10.1145/3543507.3583466](https://doi.org/10.1145/3543507.3583466) | [GitHub](https://github.com/jhliu0807/PGSP) |
| | **FaGSP** | Frequency-aware Graph Signal Processing for Collaborative Filtering | [10.1145/3701716.3715485](https://doi.org/10.1145/3701716.3715485) | [GitHub](https://github.com/Yaveng/FaGSP) |
| | **HiGSP** | Hierarchical Graph Signal Processing for Collaborative Filtering | [10.1145/3589334.3645368](https://doi.org/10.1145/3589334.3645368) | [GitHub](https://github.com/Yaveng/HiGSP) |
| **Spectral Propagation** | **GF-CF** | How Powerful is Graph Convolution for Recommendation? | [10.1145/3459637.3482387](https://doi.org/10.1145/3459637.3482387) | [GitHub](https://github.com/yshenaw/GF_CF) |
| | **BSPM** | Blurring-Sharpening Process Models for Collaborative Filtering | [10.1145/3539618.3591748](https://doi.org/10.1145/3539618.3591748) | [GitHub](https://github.com/jeongwhanchoi/BSPM) |
| | **TurboCF** | Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation | [10.1145/3626772.3657723](https://doi.org/10.1145/3626772.3657723) | [GitHub](https://github.com/jindeok/Turbo-CF) |
| | **ChebyCF** | Graph Spectral Filtering with Chebyshev Interpolation for Recommendation | [10.1145/3726302.3729991](https://doi.org/10.1145/3726302.3729991) | [GitHub](https://github.com/chanwoo0806/ChebyCF) |

---

## Our Challenges

Our work addresses four key challenges in the field:

1. **Reproducibility Assessment:** We conduct a systematic reproducibility analysis of three major families of spectral graph recommenders—spectral denoising models, graph signal processing (GSP) approaches, and spectral propagation methods. We uncover critical inconsistencies in prior evaluations, including flawed implementations of Recall and nDCG that significantly inflated reported results.

2. **Fair Benchmarking:** We establish a rigorous benchmarking framework under a unified experimental protocol with consistent data splits and extensive, comparable hyperparameter optimization for all models. We evaluate 24 methods across four public datasets, including strong classical baselines such as SLIM, Item-$k$NN, and EASE$^R$, providing a transparent and controlled comparison.

3. **Robustness to Data Sparsity:** We revisit claims that spectral denoising models are inherently more robust in sparse settings. Through controlled sparsity reduction experiments, we analyze performance degradation across varying training densities, showing that spectral approaches do not consistently outperform well-tuned traditional recommenders under extreme sparsity.

4. **Beyond-Accuracy and Complexity Analysis:** We complement accuracy evaluation with a detailed analysis of catalog coverage, novelty, and popularity bias. Our results show that while spectral filtering does not uniformly improve accuracy, it can enhance exploration and long-tail exposure, offering a nuanced perspective on the trade-offs between architectural complexity and practical recommendation benefits.

---

## Table of Contents

- [⚙️ Prerequisites & Installation](#%EF%B8%8F-prerequisites--installation)
- [🚀 Reproducing Experimental Results](#-reproducing-experimental-results)
  - [Challenge 1: Replicability Study (Table 2/3/4)](#challenge-1-replicability-study-table-234)
  - [Challenge 2: Fair Benchmarking (Table 5)](#challenge-2-fair-benchmarking-table-5)
  - [Challenge 3: Revisiting Robustness Claims (Figure 1)](#challenge-3-revisiting-robustness-claims-figure-1)
  - [Challenge 4: Assessing Architectural Complexity (Table 7)](#challenge-4-assessing-architectural-complexity-table-7)

---

## ⚙️ Prerequisites & Installation

We implemented and tested our experiments in a Python 3.8.20 environment.
For full reproducibility, we recommend using a machine with CUDA support.

1.  Create and activate a virtual environment:
```
# PYTORCH ENVIRONMENT (CUDA 12.4, cuDNN 8.9.2.26)
$ python3.8 -m venv venv
$ source venv/bin/activate
```



2.  Install the required packages from the \`requirements.txt\` file:
```
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
---























## 🚀 Reproducing Experimental Results

All experiments can be launched using the main Elliot runner script. The general command is:
```
python start_experiments.py --config <config_file_name>             # do not type ".yml" !!!!!!
```

Each research question corresponds to a set of configuration files that reproduce the results reported in the paper's tables and figures.














### Challenge 1: Replicability Study (Table 2/3/4)

To replicate the results from the original papers (Table 2/3/4, one for each models' family), run the command below.

Just replace `<dataset_name>` and `<model_name>`.

```
python start_experiments.py --config reproducibility_fpsr_<dataset_name>
```

N.B. For further details, please visit the authors' respective repositories.














### **Challenge 2**: Fair Benchmarking (Table 5)

To reproduce our main benchmark results from Table 5, you have two options.

Replace `<dataset_name>` with one of the following: `amazon_cds`, `douban`, `gowalla`, or `yelp2018`.

#### Option 1: Reproduce Final Results Directly (Recommended)

To directly obtain the results shown in Table 3, you can run the experiments using the pre-configured best hyperparameter settings for all models.
```
python start_experiments.py --config best_<dataset_name>
```

#### Option 2: Re-run the Full Hyperparameter Optimization
If you wish to replicate the entire hyperparameter search process (20 TPE trials for each model), you can use the exploration configuration files. **Note: This process is computationally expensive.**
```
python start_experiments.py --config explorations_<dataset_name>
```

N.B. In the folder `recs_SIGIR_26`, we share all the recommendation lists needed for the also all the `significativity_<dataset_name>` experiments.









### **Challenge 3**: Revisiting Robustness Claims (Figure 1)

For the robustness-to-sparsity analysis, we conduct controlled experiments on Douban by progressively reducing the training density while keeping the test set fixed (20% of interactions).
To obtain numerical results, you can use the following commands:
```
python start_experiments.py --config sparsity_douban_0_2
python start_experiments.py --config sparsity_douban_0_4
python start_experiments.py --config sparsity_douban_0_6
```















### **Challenge 4**: Assessing Architectural Complexity (Table 7)

To reproduce the fine-grained long-tail analysis results from Table 7, you can use the following commands:

```
python start_experiments.py --config beyond_accuracy_gowalla
python start_experiments.py --config beyond_accuracy_yelp2018
```












---


## 👥 Authors
- Domenico de Gioia (domenico.degioia@poliba.it)
- Claudio Pomo (claudio.pomo@poliba.it)
- Ludovico Boratto (ludovico.boratto@unica.it)
- Tommaso Di Noia (tommaso.dinoia@poliba.it)


---





## ✒️ Citation

If you find this work useful for your research, please cite our paper:
```
TBA
```
