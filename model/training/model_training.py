import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load the wine dataset
red = pd.read_csv("data/red_wine.csv")   # Replace with the actual path to the red wine data
white = pd.read_csv("data/white_wine.csv")  # Replace with the actual path to the white wine data

# Add `type` column: 1 for red, 0 for white
red['type'] = 1
white['type'] = 0

# Concatenate the datasets
wines = pd.concat([red, white], ignore_index=True)

# Feature list
features = ['fixed acidity',
            'volatile acidity',
            'citric acid',
            'residual sugar',
            'chlorides',
            'free sulfur dioxide',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol']

# Split features (X) and target (y)
X = wines[features].values
y = wines['type'].values        # `type` column is the target

# Train-test split - ensure balanced classes of red and white wines
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=45)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the PyTorch model
class WineModel(nn.Module):
    def __init__(self):
        super(WineModel, self).__init__()
        self.fc1 = nn.Linear(11, 12)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(12, 9)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(9, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def main():
    # Instantiate the model
    model = WineModel()

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    # Testing showed that 150 epochs are sufficient for this model
    epochs = 150
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Model evaluation on test data
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == y_test).float().mean().item()

    print(f"Test Loss: {test_loss.item()}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), "model/wine_model.pth")
    print("Model trained and saved as 'wine_model.pth'.")

