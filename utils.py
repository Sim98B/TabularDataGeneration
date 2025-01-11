import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, iqr, entropy

import matplotlib.pyplot as plt
import seaborn as sns

def describe_data(data: pd.DataFrame, target_col: str):

  '''
  Produces a report with descriptive statistics of the data: mean, standard deviation, minimum, maximum, skewness, curiosities, and interquartile range. 
  Useful for numerical comparison between actual and generated data.
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

def plot_data(data1: pd.DataFrame, class_var: str, data2: pd.DataFrame = None):
    
    '''
    Plot the distributions and characteristics of the data.
    If only one dataset is provided it plots the characteristics of the data; 
    if two are provided it shows the differences between and characteristics of the two datasets.
    
    Args:
    data1: real data dataset
    target: pd.Series used to plot characteristics within classes
    data2: fake or generated data dataset
    '''
    
    if data2 is None:
        fig, ax = plt.subplots(1, len(data1.select_dtypes('number').columns), figsize = (16, 4))
        for idx, feature in enumerate(data1.select_dtypes('number').columns):
            sns.boxplot(data = data1, x = feature, hue = class_var, ax = ax[idx])
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
            ax[idx].set_xlabel('')
            ax[idx].set_title(data1.select_dtypes('number').columns[idx].capitalize(), weight = 'bold')
            if idx != 0:
                ax[idx].legend().remove()    
        plt.tight_layout();
        
        fig, ax = plt.subplots(1, len(data1.select_dtypes('number').columns), figsize = (16, 4))
        for idx, feature in enumerate(data1.select_dtypes('number').columns):
            sns.kdeplot(data = data1, x = feature, hue = class_var, fill = True, alpha = 0.6, label = class_var, ax = ax[idx])
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
            ax[idx].set_xlabel('')
            ax[idx].set_ylabel('')
            if idx != 0:
                ax[idx].legend().remove()    
        plt.tight_layout();
        
        plt.figure(figsize = (8, 8))
        sns.heatmap(data = data1.select_dtypes('number').corr(), cmap = 'seismic', vmin = -1, vmax = 1, annot = True, fmt = '.3f', cbar_kws = {'location':'top'})
        plt.tight_layout();
        
        fig, ax = plt.subplots(1, data1[class_var].nunique(), figsize = (12, 4))
        for idx, target in enumerate(data1[class_var].unique()):
            sns.heatmap(data = data1[data1[class_var] == target].select_dtypes('number').corr(), cmap = 'seismic', vmin = -1, vmax = 1, annot = True, fmt = '.3f', cbar = False, ax = ax[idx])
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
            ax[idx].set_xlabel('')
            ax[idx].set_ylabel('')
            ax[idx].set_title(target.capitalize(), weight = 'bold')
        plt.tight_layout();
        
        
    else:
        
        data1 = data1.copy()
        data2 = data2.copy()
        data1['label'] = 'Real'
        data2['label'] = 'Fake'
        combined_data = pd.concat([data1, data2], ignore_index = True)
        
        fig, ax = plt.subplots(1, len(combined_data.select_dtypes('number').columns), figsize = (16, 4))
        for idx, feature in enumerate(combined_data.select_dtypes('number').columns):
            sns.boxplot(data = combined_data, x = feature, hue = 'label', ax = ax[idx])
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
            ax[idx].set_xlabel('')
            ax[idx].set_title(combined_data.select_dtypes('number').columns[idx].capitalize(), weight = 'bold')
            if idx != 0:
                ax[idx].legend().remove()    
        plt.tight_layout();
        
        fig, ax = plt.subplots(1, len(combined_data.select_dtypes('number').columns), figsize = (16, 4))
        for idx, feature in enumerate(combined_data.select_dtypes('number').columns):
            sns.kdeplot(data = combined_data, x = feature, hue = 'label', fill = True, alpha = 0.6, label = 'label', ax = ax[idx])
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
            ax[idx].set_xlabel('')
            ax[idx].set_ylabel('')
            if idx != 0:
                ax[idx].legend().remove()    
        plt.tight_layout();
        
        data1corr = combined_data[combined_data['label'] == 'Real'].select_dtypes('number').corr()
        data2corr = combined_data[combined_data['label'] == 'Fake'].select_dtypes('number').corr()
        plt.figure(figsize = (8, 8))
        sns.heatmap(data = (data1corr - data2corr), cmap = 'seismic', vmin = -1, vmax = 1, annot = True, fmt = '.3f', cbar_kws = {'location':'top'})
        plt.tight_layout();
        
        fig, ax = plt.subplots(1, combined_data[class_var].nunique(), figsize = (12, 4))
        for idx, target in enumerate(combined_data[class_var].unique()):
            
            real_corr_data = combined_data[(combined_data['target'] == target) & (combined_data['label'] == 'Real')].select_dtypes('number').corr()
            fake_corr_data = combined_data[(combined_data['target'] == target) & (combined_data['label'] == 'Fake')].select_dtypes('number').corr()
            
            sns.heatmap(data = (real_corr_data - fake_corr_data), cmap = 'seismic', vmin = -1, vmax = 1, annot = True, fmt = '.3f', cbar = False, ax = ax[idx])
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
            ax[idx].set_xlabel('')
            ax[idx].set_ylabel('')
            ax[idx].set_title(target.capitalize(), weight = 'bold')
        plt.tight_layout();
            
            
            
plot_data(data1 = real_iris, class_var = 'target', data2 = df)
