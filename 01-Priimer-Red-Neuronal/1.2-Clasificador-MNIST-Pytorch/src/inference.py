import torch  # Importa PyTorch
import torchvision.transforms as transforms  # Importa las transformaciones necesarias

def predict_image(model, image):  # Define una función para hacer inferencia en una sola imagen
    transform = transforms.Compose([  # Aplica las mismas transformaciones que durante el entrenamiento
        transforms.ToTensor(),  # Convierte la imagen a un tensor
        transforms.Normalize((0.5,), (0.5,))  # Normaliza la imagen
    ])
    
    image = transform(image).unsqueeze(0)  # Transforma y añade una dimensión para el batch (batch_size=1)
    
    model.eval()  # Coloca el modelo en modo evaluación (importante para desactivar dropout, etc.)
    with torch.no_grad():  # Desactiva el cálculo de gradientes
        output = model(image)  # Realiza una pasada hacia adelante con la imagen
        _, predicted = torch.max(output, 1)  # Obtiene el índice de la clase predicha
    return predicted.item()  # Devuelve la clase predicha como un entero
