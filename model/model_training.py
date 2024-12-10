import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# Load the wine dataset
red = pd.read_csv("data/red_wine.csv")   # Replace with the actual path to the red wine data
white = pd.read_csv("data/white_wine.csv")  # Replace with the actual path to the white wine data

# Add the 'type' column
red['type'] = 1  # Red wine
white['type'] = 0  # White wine

# Combine the datasets
wines = pd.concat([red, white], ignore_index=True)

# Split features (X) and target (y)
X = wines.iloc[:, :-1].values  # Assuming all but the last column are features
y = wines['type'].values  # Target is the 'type' column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=45)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Reshape to match PyTorch expectations
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the model
class WineModel(nn.Module):
    def __init__(self):
        super(WineModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 12)
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

# Initialize the model, loss function, and optimizer
model = WineModel()
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 200  # Adjust as needed
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), "wine_model.pth")
print("Model trained and saved as 'wine_model.pth'.")

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    predicted_labels = (predictions > 0.5).float()
    accuracy = (predicted_labels == y_test).float().mean().item()

print(f"Test Loss: {test_loss.item()}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

