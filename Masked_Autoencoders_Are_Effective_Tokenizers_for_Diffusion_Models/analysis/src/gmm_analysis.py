from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import trange
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import trange
import argparse
import os
import numpy as np
from tqdm import tqdm
import h5py
import random
import torch.nn.functional as F
import os
import random
import numpy as np
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class GMM(nn.Module):
    def __init__(self, n_components, n_features):
        super(GMM, self).__init__()
        self.n_components = n_components
        self.n_features = n_features

        # Initialize the weights, means, and covariances
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)  # Mixing coefficients
        self.means = nn.Parameter(torch.randn(n_components, n_features))  # Means of Gaussians
        self.log_covariances = nn.Parameter(torch.zeros(n_components, n_features))  # Log-diagonal covariances

    def forward(self, x):
        # Calculate the log-likelihood for each data point and component
        likelihoods = []
        for i in range(self.n_components):
            diag_cov = torch.exp(self.log_covariances[i])  # Ensure positive covariance
            dist = MultivariateNormal(self.means[i], torch.diag(diag_cov))
            likelihoods.append(dist.log_prob(x))  # Use log_prob directly

        likelihoods = torch.stack(likelihoods, dim=-1)  # Shape: [n_samples, n_components]

        # Weighted sum of log-likelihoods
        weighted_log_likelihoods = likelihoods + torch.log(self.weights + 1e-10)  # Avoid log(0)
        total_likelihood = torch.logsumexp(weighted_log_likelihoods, dim=-1)  # Log-sum-exp trick

        return total_likelihood.mean()  # Return average log-likelihood

    def bic(self, x):
        n_samples = x.size(0)
        n_params = self.n_components * (1 + self.n_features + self.n_features)  # Adjust for diagonal covariance
        log_likelihood = self.forward(x).item() * n_samples
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        return bic

def save_component_statistics(model, n_components, output_path):
    # Prepare to save the data in a CSV
    component_stats = []
    min_distance = float('inf')  # Initialize min_distance to infinity

    # Calculate min ||means[i] - means[j]|| for i != j
    for i in range(n_components):
        for j in range(i + 1, n_components):  # Only compute for i < j to avoid redundant calculations
            distance = torch.norm(model.means[i] - model.means[j]).item()
            min_distance = min(min_distance, distance)

    # Collect component statistics
    for i in range(n_components):
        mean_norm = torch.norm(model.means[i]).item()
        cov_norm = torch.norm(torch.exp(model.log_covariances[i])).item()  # for diagonal covariance
        mu = model.means[i].detach().cpu().numpy()
        cov_matrix = torch.exp(model.log_covariances[i]).detach().cpu().numpy()

        # Calculate second moment E(x^2)
        E_x2 = np.linalg.norm(mu)**2 + np.sum(cov_matrix) if cov_matrix.ndim == 1 else np.linalg.norm(mu)**2 + np.trace(cov_matrix)

        # Append the statistics, including min_distance for each component
        component_stats.append([
            n_components,
            i + 1,  # Cluster index starting from 1
            mean_norm,
            cov_norm,
            E_x2,
            min_distance  # Same min_distance for all rows of the same n_components
        ])

    # Convert the list of statistics to a DataFrame
    df = pd.DataFrame(component_stats, columns=[
        'n_components', 'cluster', 'mean_norm', 'cov_norm', 'second_moment', 'min_mean_distance'
    ])

    # Check if the output file already exists
    if os.path.exists(output_path):
        # If the file exists, append to it (without writing the header)
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        # If the file doesn't exist, write a new file (with header)
        df.to_csv(output_path, index=False)

def train_gmm(train_data, val_data, n_components, batch_size, n_iter, lr, device, output_path, num_workers=4):
    n_features = train_data.shape[1]
    model = GMM(n_components, n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = TensorDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_losses, val_losses = [], []

    for local_epoch in trange(n_iter):
        model.train()
        batch_train_losses = []
        for batch_data in train_dataloader:
            batch = batch_data[0].to(device)
            optimizer.zero_grad()
            loss = -model(batch)  # Maximize log-likelihood
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        epoch_train_loss = np.mean(batch_train_losses)
        train_losses.append(epoch_train_loss)

        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for batch_data in val_dataloader:
                batch = batch_data[0].to(device)
                loss = -model(batch)  # Maximize log-likelihood
                batch_val_losses.append(loss.item())
        epoch_val_loss = np.mean(batch_val_losses)
        val_losses.append(epoch_val_loss)

    save_component_statistics(model, n_components, output_path)

    return model, train_losses, val_losses

def calculate_metric(train_data, val_data, components_list, batch_size, n_iter, lr, device, output_path):
    bics = []
    final_train_losses, final_val_losses = [], []

    for n_components in components_list:
        # Train GMM
        model, train_losses, val_losses = train_gmm(train_data, val_data, n_components, batch_size, n_iter, lr, device, output_path)

        # Compute BIC
        bic = model.bic(train_data.to(device))
        bics.append(bic)

        # Store the final loss (last epoch's loss) for this n_components
        final_train_losses.append(train_losses[-1])
        final_val_losses.append(val_losses[-1])

    return bics, final_train_losses, final_val_losses
