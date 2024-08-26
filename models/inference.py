import torch
import torch.nn as nn
import json

class StockModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(StockModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def model_fn(model_dir, model_name):
    model_path = f"{model_dir}/{model_name}"
    model = StockModel(input_size=996, hidden_size1=64, hidden_size2=32, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        data = json.loads(request_body)
        return torch.tensor(data, dtype=torch.float32)

def predict_fn(input_data, model):
    with torch.no_grad():
        prediction = model(input_data)
    return prediction.item()

def output_fn(prediction, content_type):
    return json.dumps({'prediction': prediction})
