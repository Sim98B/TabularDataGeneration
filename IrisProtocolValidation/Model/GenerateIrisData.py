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
from sklearn.preprocessing import StandardScaler

def create_generator(path_to_weights: str):
    class WGenerator(nn.Module):
        def __init__(self, noise_dim, hidden_dim, class_dim, output_dim):
            super(WGenerator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(noise_dim + class_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim), 
            )
        
        def forward(self, noise, labels):
            labels = labels.unsqueeze(1)
            x = torch.cat([noise, labels], dim=1)
            return self.model(x)
        
    generator = WGenerator(noise_dim = 16, hidden_dim = 32, class_dim = 1, output_dim = 6)
    generator.load_state_dict(torch.load(path_to_weights))
    generator.eval()
    return generator

def generate_iris_data(model: torch.nn.Module, scaler, setosa: int = 50, versicolor: int = 50, virginica: int = 50, seed: int = None):
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    mapping_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    feature_names_list = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    setosa_label = torch.zeros(setosa, dtype=torch.long)
    versicolor_label = torch.ones(versicolor, dtype=torch.long)
    virginica_label = torch.full((virginica,), 2, dtype=torch.long)
    
    label_array = np.concatenate((setosa_label, versicolor_label, virginica_label))
    label_tensor = torch.cat([setosa_label, versicolor_label, virginica_label], 0)
    noise_vector = torch.randn(setosa + versicolor + virginica, 16)
    
    generated_array = model(noise_vector, label_tensor).detach().numpy()
    data_array = scaler.inverse_transform(generated_array)[:, :4]
    
    iris_dataframe = pd.DataFrame(data_array, columns=feature_names_list)
    iris_dataframe['target'] = label_array
    iris_dataframe['target'] = iris_dataframe['target'].map(mapping_dict)
    
    return iris_dataframe

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Iris dataset")
    parser.add_argument("--weights", type=str, required=True, help="Path to generator weights (.pth file)")
    parser.add_argument("--scaler", type=str, required=True, help="Path to saved scaler (.pkl file)")
    parser.add_argument("--setosa", type=int, default=50, help="Number of setosa samples")
    parser.add_argument("--versicolor", type=int, default=50, help="Number of versicolor samples")
    parser.add_argument("--virginica", type=int, default=50, help="Number of virginica samples")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="SyntheticIris.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    
    model = create_generator(args.weights)
    scaler = joblib.load(args.scaler)
    df = generate_iris_data(model, scaler, args.setosa, args.versicolor, args.virginica, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Dataset saved to {args.output}")

if __name__ == "__main__":
    main()

