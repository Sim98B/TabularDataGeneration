# Breast Cancer Tabular Data Generation
This project is the first of two sections related to the generation of **synthetic medical data** for applications in **Machine Learning** and **Data Science**. The generation of medical data presents peculiar and delicate critical issues that must be handled with caution.
The goal of this project is to train a **Wasserstein Generative Adversarial Network** (**WGAN**) capable of capturing and reproducing features of the [UCI Machine Learning Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) dataset such that the data generated:
- are as similar as possible to the real thing
- are indistinguishable for a classifier trained on the real data (its performance is equal on both datasets)

![Classifier Comparison](images/final_confusion_matrix.png)
*Classifier comparison*

---

## Table of Contents
1. [Libraries](#libraries)
2. [Data Preparation](#data-preparation)
3. [Data Modeling](#Data-modeling)


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

The dataset contains 10 features related to cancer cell measurements:
1. **Radius**: value of the average distance from the center to the points on the core perimeter.
2. **Texture**: variation of intensity in adjacent pixels.
3. **Perimeter**: total length of the core contour.
4. **Area**: area occupied by the cell nucleus.
5. **Complexity**: uniformity of the contours of the nucleus.
6. **Compacteness**: relationship between area and perimeter of the core.
7. **Concavity**: gravity of indentations along the perimeter of the core.
8. **Number of concavities**: number of concave points along the core boundary.
9. **Symmetry**: similarity between the two halves of the core.
10. **Fractal Dimension**: measure of the complexity of the shape of the core based on the fractal dimension.

**`Mean`**, **`standard deviation`**, and **`worst measure`** are reported for each feature.
Of the 569 measurements 357 are of **benign** tumors and 212 of **malignant** tumors.

The entire dataset was loaded and preprocessed:
- numerical features were scaled between **`[-1, 1]`** to promote model convergence via the **`hyperbolic tangent`** function.
- class **labels were encoded** and provided to the model to condition the generated data.

---

## Data Modeling

The two models of the **WGAN** were created following the best practices of the case and seeking some consistency with the number of instances in the dataset `(17070)`:
- **Generator**: `23.646k` parameters
- **Critic**: `26.753k` parameters

In this way we were able to:
- **adequately capture** the complex and interconnected features of the dataset
- **avoid overfitting** by having slightly more parameters than the number of cases
- **obtain quality gradients** due to a more powerful generator critic

The **Kolomonorv-Smirnov test** was used to evaluate the null hypothesis that the distributions of each real and synthetic feature were from the same population and the **Wasserstein distance** to calculate the mathematical distance between each real-fact pair of distributions:
- **KS test**: for obi feature recreated the null hypothesis was not refutable, all features generated come from the same population as the real ones.
- **W distance**: for 5 features the distance from the corresponding real distributions was found to be greater than 1.

| Feature       | W Distance    |
|--------------|--------|
| **mean perimeter** | 1.236084   |
| **worst perimeter** | 1.615902 |
| **area error** | 2.506679   |
| **mean area** | 16.915316   |
| **worst area** | 29.057681   |
