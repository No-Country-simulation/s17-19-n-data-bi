import torch
import torch.nn as nn
from sagemaker_inference import content_types, decoder, encoder, default_inference_handler
import json

class StockModel(nn.Module):
    def __init__(self):
        super(StockModel, self).__init__()
        self.fc1 = nn.Linear(996, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def model_fn(model_dir):
    model = StockModel()
    model.load_state_dict(torch.load(model_dir + '/stock_verification_model.pth'))
    model.eval()
    return model

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        data = json.loads(request_body)
        return torch.tensor(data, dtype=torch.float32)

def predict_fn(input_data, model):
    with torch.no_grad():
        return model(input_data).item()
