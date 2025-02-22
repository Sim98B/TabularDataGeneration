#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler

def create_generator(path_to_weights: str):
    class WGenerator(nn.Module):
        def __init__(self, noise_dim, class_dim, output_dim):
            super(WGenerator, self).__init__()

            self.model = nn.Sequential(
                nn.Linear(noise_dim + class_dim, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, output_dim), 
                nn.Tanh()
            )

        def forward(self, noise, labels):
            labels = labels.unsqueeze(1)
            x = torch.cat([noise, labels], dim = 1)
            return self.model(x)
        
    generator = WGenerator(noise_dim = 100, class_dim = 1, output_dim = 30)
    generator.load_state_dict(torch.load(path_to_weights))
    generator.eval()
    return generator

def generate_breast_cancer_data(model: torch.nn.Module, scaler, benign: int = 357, malignant: int = 212, seed: int = None):
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    mapping_dict = {0: 'benign', 1: 'malignant'}
    feature_names_list = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
                          'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 
                          'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 
                          'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 
                          'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 
                          'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 
                          'worst symmetry', 'worst fractal dimension']
    
    benign_label = torch.zeros(benign, dtype = torch.long)
    malignant_label = torch.ones(malignant, dtype = torch.long)
    
    label_array = np.concatenate((benign_label, malignant_label))
    label_tensor = torch.cat([benign_label, malignant_label], 0)
    noise_vector = torch.randn(benign + malignant, 100)
    
    generated_array = model(noise_vector, label_tensor).detach().numpy()
    data_array = scaler.inverse_transform(generated_array)
    
    breast_cancer_dataframe = pd.DataFrame(data_array, columns = feature_names_list)
    breast_cancer_dataframe['target'] = label_array
    breast_cancer_dataframe['target'] = breast_cancer_dataframe['target'].map(mapping_dict)
    
    return breast_cancer_dataframe

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Iris dataset")
    parser.add_argument("--weights", type = str, required = True, help = "Path to generator weights (.pth file)")
    parser.add_argument("--scaler", type = str, required = True, help = "Path to saved scaler (.pkl file)")
    parser.add_argument("--benign", type = int, default = 357, help = "Number of benign samples")
    parser.add_argument("--malignant", type = int, default = 212, help = "Number of malignant samples")
    parser.add_argument("--seed", type = int, default = None, help = "Random seed for reproducibility")
    parser.add_argument("--output", type = str, default = "SyntheticBreastCancer.csv", help = "Output CSV file path")
    
    args = parser.parse_args()
    
    model = create_generator(args.weights)
    scaler = joblib.load(args.scaler)
    df = generate_breast_cancer_data(model, scaler, args.benign, args.malignant, args.seed)
    df.to_csv(args.output, index = False)
    print(f"Dataset saved to {args.output}")

if __name__ == "__main__":
    main()

