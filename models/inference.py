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
        "marketing": None,  # Marketing se maneja a través de GenAI, no tiene un modelo entrenado en PyTorch
        "afinidad": None,   # Afinidad también se maneja con GenAI
        "demandapcbnopcb": "models/demandapcbnopcb_model.pth"
    }

    # Ajustar input_size basado en el modelo seleccionado
    if model_name == "stock":
        input_size = 1995
    elif model_name == "prevision":
        input_size = 4
    elif model_name in ["marketing", "afinidad"]:
        return None  # Para marketing y afinidad, no se carga un modelo de PyTorch
    else:
        input_size = 2  # Ajustar según las características reales de entrada

    if model_paths[model_name] is None:
        raise ValueError(f"El modelo {model_name} se maneja a través de la API GenAI, no requiere un modelo PyTorch.")

    model = BaseModel(input_size)
    if os.path.exists(model_paths[model_name]):
        # Cargar el estado del modelo
        model.load_state_dict(torch.load(model_paths[model_name], weights_only=True))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"No se encontró el archivo de modelo: {model_paths[model_name]}")

def predict(model, input_data):
    if model is None:
        raise ValueError("El modelo no está cargado correctamente.")
    
    with torch.no_grad():
        prediction = model(input_data)
    return prediction.item()

# Ejemplo de uso
# model = load_model("stock")
# pred = predict(model, some_input_tensor)
