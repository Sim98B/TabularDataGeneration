import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, iqr, entropy

import matplotlib.pyplot as plt
import seaborn as sns

def describe_data(data: pd.DataFrame, target_col: str):
  
  '''
  Provides descriptive statistics for each class, useful for comparing actual and generated data.
  
  Args:
    data (pd.DataFrame): Dataframe containing the class columns too.
    target_col (str): Name of the class columns.
  '''

  Mean = real_iris.groupby('target').mean().round(3)
  Std = real_iris.groupby('target').std().round(3)
  Min = real_iris.groupby('target').min().round(3)
  Max = real_iris.groupby('target').max().round(3)
  Skew = np.round([skew(data[data['target'] == label].iloc[:,:-1]) for label in data['target'].unique()],3)
  Kurt = np.round([kurtosis(data[data['target'] == label].iloc[:,:-1]) for label in data['target'].unique()],3)
  Iqr = np.round([[iqr(data[data['target'] == label][col]) for col in data.columns[:-1]]for label in data['target'].unique()], 3)
    
  stats = ['MEAN', 'STD', 'MIN', 'MAX', 'SKEW', 'KURT', 'IQR']
  indices_list = []
  for stat in stats:
    for i in range(len(data.columns[:-1])):
      indices_list.append((stat, features[i]))
            
  header = pd.MultiIndex.from_tuples(indices_list)
  array = np.hstack((Mean, Std, Min, Max, Skew, Kurt, Iqr))
  report = pd.DataFrame(array, columns = header, index = species).T
    
  return  report
