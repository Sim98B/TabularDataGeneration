# Breast Cancer Tabular Data Generation
This project is the first of two sections related to the generation of **synthetic medical data** for applications in **Machine Learning** and **Data Science**. The generation of medical data presents peculiar and delicate critical issues that must be handled with caution.
The goal of this project is to train a **Wasserstein Generative Adversarial Network** (**WGAN**) capable of capturing and reproducing features of the [UCI Machine Learning Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) dataset such that the data generated:
- are as similar as possible to the real thing
- are indistinguishable for a classifier trained on the real data (its performance is equal on both datasets)
