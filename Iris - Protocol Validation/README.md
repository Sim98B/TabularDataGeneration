# Iris Tabular Data Generation using PyTorch

This project explores synthetic data generation for the Iris dataset using different architectures for data generation including **Variational Auto Encoder (VAE)**, **Generative Adversarial Network (GAN)** and **Wasserstein GAN WGAN**. The goal is to generate high-quality synthetic tabular data that preserves the statistical properties and class-specific relationships of the original dataset. This work is part of a larger initiative to generate synthetic medical data (tabular and imaging) and is a validation of a protocol to be followed in data generation tasks.

![Generated vs Real Data Comparison](images/Q-Qplots.png)
*Example of synthetic vs real data distributions.*

---

## Table of Contents
1. [Libraries](#libraries)
2. [Data Preparation](#data-preparation)
3. [Exploring Architectures](#exploring-architectures)
4. [Optimizing Architecture](#optimizing-architecture)
5. [Wasserstein Conditional GAN](#introducing-the-wgan)
6. [Final Evaluation](#final-evaluation)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Results](#results)
10. [Conclusion](#conclusion)
11. [References](#references)

---

## Libraries
The project uses the following Python libraries:
- **PyTorch** for model implementation.
- **Pandas**/**NumPy** for data manipulation.
- **Scikit-learn** for metrics and preprocessing.
- **Matplotlib**/**Seaborn** for visualization.
- **SciPy** for statistical tests (Kolmogorov-Smirnov, Wasserstein distance).

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import torch
```

---

## Data Preparation
The Iris dataset is loaded and preprocessed:
- **Normalization**: Features scaled to [0, 1].
- **Class Conditioning**: Labels encoded and provided as input to generators.

---

## Exploring Architectures
### GAN vs VAE
Two architectures were compared:
1. **Variational Autoencoder (VAE)**: Learned latent representations but struggled with class-specific feature relationships.
2. **Generative Adversarial Network (GAN)**: Outperformed VAE in capturing feature characteristics as shown by images.

| AVG Metric       | GAN    | VAE    |
|--------------|--------|--------|
| Wasserstein ↓| 0.32274   | 0.25224   |
| KS Test p-value ↑ | 0.00053 | 2.5e-6 |


---
