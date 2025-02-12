import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, iqr, ks_2samp, wasserstein_distance
import random

import matplotlib.pyplot as plt
import seaborn as sns

import torch

def calculate_ks_test(real_data, synthetic_data):
    '''
    Return a pd.DataFrame with Kolmogorov-Smirnov statistic and the associated p-value for all the dataframe features.
    
    Args:
        real_data (pd.DataFrame): Dataframe containing real data
        synthetic_data (pd.DataFrame): Dataframe containing fake data
        
    Returns:
        pd.DataFrame with the result of the KS test.
    '''
    
    real_data = real_data.select_dtypes('number')
    synthetic_data = synthetic_data.select_dtypes('number')
    results = []
    for i in range(real_data.shape[1]):
        ks_stat, p_value = ks_2samp(real_data.iloc[:, i], synthetic_data.iloc[:, i])
        results.append({'Feature': real_data.columns[i], 'KS Statistic': ks_stat, 'P-Value': p_value})
    return pd.DataFrame(results)

def calculate_wasserstein_distance(real_data, synthetic_data):
    '''
    Return a pd.DataFrame with the Wasserstein Distance for all the dataframe features.
    
    Args:
        real_data (pd.DataFrame): Dataframe containing real data
        synthetic_data (pd.DataFrame): Dataframe containing fake data
        
    Returns:
        pd.DataFrame with all the distances.
    '''
    real_data = real_data.select_dtypes('number')
    synthetic_data = synthetic_data.select_dtypes('number')
    results = []
    for i in range(real_data.shape[1]):
        w_distance = wasserstein_distance(real_data.iloc[:, i], synthetic_data.iloc[:, i])
        results.append({'Feature': real_data.columns[i], 'W Distance': w_distance})
    return pd.DataFrame(results)

def compare_results(data1: torch.tensor, data2: torch.tensor):
    '''
    Utility function to monitor differences in descriptive statistics between real and fake data while training a model.
    
    Args:
        data1 (torch.tensor): Tensor of real data
        data2 (torch.tensor): Tensor of fake data
        
    Returns:
        Tensor with differences between descriptive statistics each feature
    '''
    
    mean = np.round(np.array(data1.mean(0) - data2.mean(0)), 3)
    std = np.round(np.array(data1.std(0) - data2.std(0)), 3)
    minimum = np.round(np.array(data1.min(0)[0] - data2.min(0)[0]), 3)
    maximum = np.round(np.array(data1.max(0)[0] - data2.max(0)[0]), 3)
    skewness = np.round(skew(data1) - skew(data2), 3)
    kurt = np.round(kurtosis(data1) - kurtosis(data2), 3)
    
    print(f'MEAN: {mean}\nSTD:  {std}\nMIN:  {minimum}\nMAX:  {maximum}\nSKEW: {skewness}\nKURT: {kurt}')
    
def critic_loss(critic: torch.nn.Module, real_data: torch.Tensor, fake_data: torch.Tensor, labels: torch.Tensor, lambda_gp: float = 10):
    """
    Calculates the critical loss for a WGAN-GP, including the gradient penalty.

    Args:
        critic (torch.nn.Module): The critic (discriminator) that evaluates real and generated data.
        real_data (torch.Tensor): Batch of real data with shape (batch_size, feature_dim).
        fake_data (torch.Tensor): Batch of generated data with shape (batch_size, feature_dim).
        labels (torch.Tensor): Conditional labels associated with data, if applicable.
        lambda_gp (float, optional): Coefficient that adjusts the weight of the gradient penalty. Default: 10.

    Returns:
        torch.Tensor: Scaled value of the critic's loss.
    """
    real_score = critic(real_data, labels)
    fake_score = critic(fake_data, labels)
    
    wasserstein_loss = fake_score.mean() - real_score.mean()
    
    gp = gradient_penalty(critic, real_data, fake_data, labels, lambda_gp)
    return wasserstein_loss + gp

def describe_data(data1: pd.DataFrame, target_col: str, data2: pd.DataFrame = None):
    '''
    Provides descriptive statistics for each class, useful for comparing actual and generated data.
  
    Args:
      data1 (pd.DataFrame): Dataframe containing the class columns too.
      target_col (str): Name of the class column.
      data2 (pd.DataFrame, optional): Optional dataframe of synthetic data. If provided, the final report will include
      descriptive statistics for both datasets. Default is None.
          
    Returns:
      pd.DataFrame: A dataframe with descriptive statistics for the provided datasets.
    '''
    
    def generate_stats(data, target_col):
        """
        Helper function to compute statistics for datasets
        """
        
        mean = data.groupby(target_col).mean().round(3)
        std = data.groupby(target_col).std().round(3)
        min_val = data.groupby(target_col).min().round(3)
        max_val = data.groupby(target_col).max().round(3)
        skewness = np.round([skew(data[data[target_col] == label].iloc[:, :-1]) for label in data[target_col].unique()], 3)
        kurt = np.round([kurtosis(data[data[target_col] == label].iloc[:, :-1]) for label in data[target_col].unique()], 3)
        iqr_values = np.round([[iqr(data[data[target_col] == label][col]) for col in data.columns[:-1]] 
                               for label in data[target_col].unique()], 3)

        stats = ['MEAN', 'STD', 'MIN', 'MAX', 'SKEW', 'KURT', 'IQR']
        indices_list = []
        for stat in stats:
            for i in range(len(data.columns[:-1])):
                indices_list.append((stat, data.select_dtypes('number').columns[i]))
        
        header = pd.MultiIndex.from_tuples(indices_list)
        array = np.hstack((mean, std, min_val, max_val, skewness, kurt, iqr_values))
        return pd.DataFrame(array, columns=header, index=data[target_col].unique()).T

    report1 = generate_stats(data1, target_col)
    report1.columns = pd.MultiIndex.from_product([["REAL"], report1.columns])

    if data2 is None:
        return report1
    
    else:
        report2 = generate_stats(data2, target_col)
        report2.columns = pd.MultiIndex.from_product([["SYNTHETIC"], report2.columns])

        final_report = pd.concat([report1, report2], axis=1)
        return final_report
    
def generator_loss(critic: torch.nn.Module, fake_data: torch.Tensor, labels: torch.Tensor):
    """
    Calculates the loss of the generator in a WGAN-GP.

    Args:
        critic (torch.nn.Module): The critic (discriminator) that evaluates real and generated data.
        fake_data (torch.Tensor): Batch of generated data with shape (batch_size, feature_dim).
        labels (torch.Tensor): Conditional labels associated with data, if applicable.

    Returns:
        torch.Tensor: Scaled value of the generator loss.
    """
    fake_score = critic(fake_data, labels)
    return -fake_score.mean()
    
def gradient_penalty(critic: torch.nn.Module, real_data: torch.Tensor, fake_data: torch.Tensor, labels: torch.Tensor, lambda_gp: float = 10):
    """
    Calculates the gradient penalty term for stabilizing WGAN-GP training.

    Args:
        critic (torch.nn.Module): The critic (discriminator) that evaluates real and generated data.
        real_data (torch.Tensor): Batch of real data with shape (batch_size, feature_dim).
        fake_data (torch.Tensor): Batch of generated data with shape (batch_size, feature_dim).
        labels (torch.Tensor): Conditional labels associated with data, if applicable.
        lambda_gp (float, optional): Coefficient that adjusts the weight of the gradient penalty. Default: 10.

    Returns:
        torch.Tensor: Scaled value of the gradient penalty to be added to the critical's loss.
    """
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1)
    epsilon = epsilon.expand_as(real_data)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated.requires_grad_(True)
    
    interpolated_score = critic(interpolated, labels)
    
    grad_outputs = torch.ones_like(interpolated_score)
    gradients = torch.autograd.grad(
        outputs=interpolated_score,
        inputs=interpolated,
        grad_outputs = grad_outputs,
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]
    
    gradients_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    penalty = lambda_gp * ((gradients_norm - 1) ** 2).mean()
    return penalty

def plot_data(data1: pd.DataFrame, class_var: str, data2: pd.DataFrame = None):
    
    '''
    Plot the distributions and characteristics of the data.
    If only one dataset is provided it plots the characteristics of the data; 
    if two are provided it shows the differences between and characteristics of the two datasets.
    
    Args:
        data1 (pd.DataFrame): real data dataset. 
        target (str): pd.Series used to plot characteristics within classes. 
        data2(pd.DataFrame): fake or generated data dataset. Default is None.
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
        sns.heatmap(data = data1.select_dtypes('number').corr(), cmap = 'seismic', vmin = -1, vmax = 1, annot = True, fmt = '.3f', cbar_kws = {'location':'top'}, annot_kws = {'weight': 'bold'})
        plt.tight_layout();
        
        fig, ax = plt.subplots(1, data1[class_var].nunique(), figsize = (12, 4))
        for idx, target in enumerate(data1[class_var].unique()):
            sns.heatmap(data = data1[data1[class_var] == target].select_dtypes('number').corr(), cmap = 'seismic', vmin = -1, vmax = 1, annot = True, fmt = '.3f', cbar = False, ax = ax[idx], annot_kws = {'weight': 'bold'})
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
            sns.boxplot(data = combined_data, x = feature, y = class_var, hue = 'label', ax = ax[idx])
            ax[idx].set_xticks([])
            ax[idx].set_xlabel('')
            ax[idx].set_ylabel('')
            ax[idx].set_title(combined_data.select_dtypes('number').columns[idx].capitalize(), weight = 'bold')
            if idx != 0:
                ax[idx].legend().remove()  
                ax[idx].set_yticks([])
        plt.tight_layout();
        
        for i, row in enumerate(combined_data[class_var].unique()):
            fig, ax = plt.subplots(1, len(combined_data.select_dtypes('number').columns), figsize = (16, 4))
            for idx, feature in enumerate(combined_data.select_dtypes('number').columns):
                sns.kdeplot(data = data1[data1['target'] == row], x = feature, fill = True, alpha = 0.6, label = 'Real', ax = ax[idx])
                sns.kdeplot(data = data2[data2['target'] == row], x = feature, fill = True, alpha = 0.6, label = 'Fake', ax = ax[idx])
                ax[idx].set_xticks([])
                ax[idx].set_yticks([])
                ax[idx].set_xlabel('')
                ax[idx].set_ylabel(row.capitalize(), weight = 'bold')
                ax[idx].set_title(feature.capitalize(), weight = 'bold')
                ax[idx].legend(loc = 'upper right')
                if idx != 0:
                    ax[idx].legend().remove()
                    ax[idx].set_ylabel('')
                if i != 0:
                    ax[idx].set_title(None)
        plt.tight_layout();
        
        data1corr = combined_data[combined_data['label'] == 'Real'].select_dtypes('number').corr()
        data2corr = combined_data[combined_data['label'] == 'Fake'].select_dtypes('number').corr()
        plt.figure(figsize = (8, 8))
        sns.heatmap(data = (data1corr - data2corr), cmap = 'seismic', vmin = -1, vmax = 1, annot = True, fmt = '.3f', cbar_kws = {'location':'top'}, annot_kws = {'weight': 'bold'})
        plt.tight_layout();
        
        fig, ax = plt.subplots(1, combined_data[class_var].nunique(), figsize = (12, 4))
        for idx, target in enumerate(combined_data[class_var].unique()):
            
            real_corr_data = combined_data[(combined_data['target'] == target) & (combined_data['label'] == 'Real')].select_dtypes('number').corr()
            fake_corr_data = combined_data[(combined_data['target'] == target) & (combined_data['label'] == 'Fake')].select_dtypes('number').corr()
            
            sns.heatmap(data = (real_corr_data - fake_corr_data), cmap = 'seismic', vmin = -1, vmax = 1, annot = True, fmt = '.3f', cbar = False, annot_kws = {'weight': 'bold'}, ax = ax[idx])
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
            ax[idx].set_xlabel('')
            ax[idx].set_ylabel('')
            ax[idx].set_title(target.capitalize(), weight = 'bold')
        plt.tight_layout();
        
def plot_quantiles(data1: pd.DataFrame, data2: pd.DataFrame):
    
    '''
    Shows a grid with Q-Q plots of real and fake data to compare distributions.
    
    Args:
        data1 (pd.DataFrame): real dataset
        data2 (pd.DataFrame): fake dataset
        
    Return:
        Grid with Q-Q plots.
    '''
    
    quantiles = np.linspace(0, 1, len(data1))
    
    fig, ax = plt.subplots(1, data1.select_dtypes('number').shape[1], figsize = (16, 4))
    
    for idx, col in enumerate(data1.select_dtypes('number').columns):
        
        real_data = np.sort(data1[col])
        fake_data = np.sort(data2[col])
        
        real_quantiles = np.percentile(real_data, quantiles * 100)
        fake_quantiles = np.percentile(fake_data, quantiles * 100)
        
        ax[idx].scatter(real_quantiles, fake_quantiles)
        ax[idx].plot([real_quantiles.min(), real_quantiles.max()], 
                     [real_quantiles.min(), real_quantiles.max()], 
                     color='red', linestyle='--', label='Expected Line')
        ax[idx].set_xlabel('Quantiles of Real Data')
        ax[idx].set_ylabel('Quantiles of Synthetic Data')
        ax[idx].set_title('Q-Q Plot')
        ax[idx].legend()
        ax[idx].grid(True)
        plt.tight_layout();
        
def set_seed(seed: int = 0):
    '''
    Sets a seed for all random operations to ensure reproducible results.
    
    Args:
        seed (int): The seed value to be used for random number generation. Default is 0.
    '''
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)