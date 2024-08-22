import torch  # Importa PyTorch
import torchvision  # Importa el módulo de visión por computadora de PyTorch
import torchvision.transforms as transforms  # Importa transformaciones para los datos

def get_data_loaders(batch_size=4):  # Define una función para obtener los cargadores de datos
    transform = transforms.Compose([  # Define las transformaciones a aplicar a los datos
        transforms.ToTensor(),  # Convierte las imágenes a tensores
        transforms.Normalize((0.5,), (0.5,))  # Normaliza los datos con media 0.5 y desviación estándar 0.5
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # Carga el conjunto de datos MNIST para entrenamiento
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  # Crea un DataLoader para el conjunto de entrenamiento
    
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # Carga el conjunto de datos MNIST para prueba
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)  # Crea un DataLoader para el conjunto de prueba
    
    return train_loader, test_loader  # Devuelve los DataLoaders para entrenamiento y prueba