# Iris Tabular Data Generation using PyTorch

This project explores synthetic data generation for the Iris dataset using **Conditional Generative Adversarial Networks (cGANs)** and **Variational Autoencoders (VAEs)**. The goal is to generate high-quality synthetic tabular data that preserves the statistical properties and class-specific relationships of the original dataset. This work is part of a broader initiative to generate synthetic medical data (tabular and imaging).

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
