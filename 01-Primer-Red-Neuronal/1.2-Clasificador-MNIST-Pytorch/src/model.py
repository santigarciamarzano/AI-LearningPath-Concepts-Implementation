import torch.nn as nn  # Importa el módulo de redes neuronales de PyTorch
import torch.nn.functional as F  # Importa funciones de activación y otras utilidades

class Net(nn.Module):  # Define una nueva clase para el modelo que hereda de nn.Module
    def __init__(self):  # Inicializa la clase
        super(Net, self).__init__()  # Llama al constructor de la clase base
        self.fc1 = nn.Linear(28 * 28, 500)  # Capa lineal de entrada con 28*28 neuronas de entrada y 500 neuronas ocultas
        self.fc2 = nn.Linear(500, 10)  # Capa lineal de salida con 500 neuronas de entrada y 10 neuronas de salida

    def forward(self, x):  # Define la función de paso hacia adelante
        x = x.view(-1, 28 * 28)  # Aplana la imagen (cambia la forma de [batch_size, 1, 28, 28] a [batch_size, 28*28])
        x = F.relu(self.fc1(x))  # Aplica la función de activación ReLU a la salida de la primera capa
        x = self.fc2(x)  # Pasa el resultado a la segunda capa
        return x  # Devuelve la salida final
