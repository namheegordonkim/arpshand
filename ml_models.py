import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn as nn
from tqdm import tqdm

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


class SequenceAutoencoderClassifier:

    def __init__(self, in_dimension, embedding_dimension, out_dimension, hidden_dimensions):
        self.in_dimension = in_dimension
        self.embedding_dimension = embedding_dimension
        self.out_dimension = out_dimension
        self.hidden_dimensions = hidden_dimensions

        self.classifier_linear = nn.Linear(self.embedding_dimension, self.out_dimension)
        self.encoder = MultiLayerPerceptron(self.in_dimension, self.embedding_dimension, self.hidden_dimensions)
        self.decoder = MultiLayerPerceptron(self.embedding_dimension, self.in_dimension, self.hidden_dimensions)
        self.scaler = StandardScaler()

    def fit(self, X, y, num_iter=10000):
        n, d = X.shape
        k = len(np.unique(y))
        X_scaled = self.scaler.fit_transform(X)

        learnable_parameters = []
        learnable_parameters.extend(self.classifier_linear.parameters())
        learnable_parameters.extend(self.encoder.parameters())
        optimizer = torch.optim.Adam(lr=1e-4, params=learnable_parameters)

        cross_entropy = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        l1_loss = nn.L1Loss()

        self.encoder.to(device)
        self.decoder.to(device)
        self.classifier_linear.to(device)
        for i in tqdm(range(num_iter)):
            X_tensor = torch.as_tensor(X_scaled).float().to(device)
            y_tensor = torch.as_tensor(y).long().to(device)
            Z_tensor = self.encoder.forward(X_tensor)
            Xhat_tensor = self.decoder.forward(Z_tensor)
            yhat_tensor = self.classifier_linear.forward(Z_tensor)

            cross_entropy_loss = cross_entropy.forward(yhat_tensor, y_tensor)
            autoencoder_loss = l1_loss.forward(Xhat_tensor, X_tensor)
            loss = 1e-3 * cross_entropy_loss + autoencoder_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print("Loss: {:f}".format(loss))

        self.encoder.cpu()
        self.decoder.cpu()
        self.classifier_linear.cpu()

    def transform(self, X):
        n, d = X.shape
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.as_tensor(X_scaled).float()
        Z_tensor = self.encoder.forward(X_tensor)
        Z = Z_tensor.cpu().detach().numpy()
        return Z

    def inverse_transform(self, Z):
        Z_tensor = torch.as_tensor(Z).float()
        X_tensor = self.decoder.forward(Z_tensor)
        X_scaled = X_tensor.cpu().detach().numpy()
        X = self.scaler.inverse_transform(X_scaled)
        return X


class MultiLayerPerceptron(nn.Module):

    def __init__(self, in_dimension, out_dimension, hidden_dimensions):
        super().__init__()
        n_hidden_layers = len(hidden_dimensions)
        self.fcs = nn.ModuleList()
        fc = nn.Linear(in_dimension, hidden_dimensions[0])
        self.fcs.append(fc)
        for i in range(n_hidden_layers - 1):
            fc = nn.Linear(hidden_dimensions[i], hidden_dimensions[i + 1])
            self.fcs.append(fc)
        fc = nn.Linear(hidden_dimensions[-1], out_dimension)
        self.fcs.append(fc)
        self.relu = nn.ReLU()

    def forward(self, input):
        for fc in self.fcs[:-1]:
            input = fc.forward(input)
            input = self.relu(input)
        input = self.fcs[-1].forward(input)
        return input
