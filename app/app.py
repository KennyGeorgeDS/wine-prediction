import os

import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel


# Define the input schema with actual field names
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# Define the model architecture (same as the training script)
class WineModel(nn.Module):
    def __init__(self):
        super(WineModel, self).__init__()
        self.fc1 = nn.Linear(11, 12)  # Input layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(12, 9)  # Hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(9, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

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
model_path = os.path.join(os.path.dirname(__file__), "../model/wine_model.pth")  # Adjust path
model.load_state_dict(torch.load(model_path))
model.eval()

# Initialize FastAPI app
app = FastAPI()


@app.post("/predict")
async def predict(input_data: WineInput):
    """
    Predict the wine type (red or white) based on individual feature inputs.
    """
    # Convert individual fields into a PyTorch tensor
    input_tensor = torch.tensor(
        [
            [
                input_data.fixed_acidity,
                input_data.volatile_acidity,
                input_data.citric_acid,
                input_data.residual_sugar,
                input_data.chlorides,
                input_data.free_sulfur_dioxide,
                input_data.total_sulfur_dioxide,
                input_data.density,
                input_data.pH,
                input_data.sulphates,
                input_data.alcohol,
            ]
        ],
        dtype=torch.float32,
    )

    # Perform prediction
    prediction = model(input_tensor).item()

    # Determine wine type
    wine_type = "Red" if prediction > 0.5 else "White"

    return {"prediction": prediction, "wine_type": wine_type}  # Probability output by the model  # Interpreted result
