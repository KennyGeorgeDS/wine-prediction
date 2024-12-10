from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import os

# Define the input schema
class WineInput(BaseModel):
    features: list[float]  # Input features as a list of floats

# Define the model architecture (same as used in training)
class WineModel(nn.Module):
    def __init__(self):
        super(WineModel, self).__init__()
        self.fc1 = nn.Linear(11, 12)  # Adjust input size if necessary
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

# Load the trained model
model = WineModel()
model_path = os.path.join("model", "wine_model.pth")  # Path to the model
model.load_state_dict(torch.load(model_path))
model.eval()

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(input_data: WineInput):
    # Convert input features to a PyTorch tensor
    input_tensor = torch.tensor([input_data.features], dtype=torch.float32)
    # Get the model's prediction
    prediction = model(input_tensor).item()
    # Convert the prediction to a class label
    wine_type = "Red" if prediction > 0.5 else "White"
    return {
        "prediction": prediction,
        "wine_type": wine_type
    }

