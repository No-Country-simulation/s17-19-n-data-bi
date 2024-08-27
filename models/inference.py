import torch
import torch.nn as nn
import os

# Definición de la estructura del modelo, común para todos los casos
class BaseModel(nn.Module):
    def __init__(self, input_size):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def load_model(model_name):
    model_paths = {
        "stock": "models/stock_model.pth",
        "prevision": "models/prevision_model.pth",
        "marketing": None,
        "afinidad": None,
        "demandapcbnopcb": "models/demandapcbnopcb_model.pth"
    }

    input_size_map = {
        "stock": 1995,
        "prevision": 4,
        "demandapcbnopcb": 2
    }
    
    input_size = input_size_map.get(model_name)
    if input_size is None and model_name not in ["marketing", "afinidad"]:
        raise ValueError(f"Modelo desconocido o sin input_size definido: {model_name}")

    if model_paths[model_name] is None:
        raise ValueError(f"El modelo {model_name} se maneja a través de la API GenAI, no requiere un modelo PyTorch.")

    model = BaseModel(input_size)
    model_path = model_paths[model_name]
    
    if os.path.exists(model_path):
        # Cargar el estado del modelo
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Cambiar el modelo a modo de evaluación
        return model
    else:
        raise FileNotFoundError(f"No se encontró el archivo de modelo: {model_path}")
