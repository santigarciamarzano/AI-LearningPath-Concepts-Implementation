from PIL import Image
import torch
import numpy as np
from src.model import Net  # Asegúrate de importar la clase Net desde el módulo correcto

def load_model(model_path):
    model = Net()  # inicializa el modelo
    model.load_state_dict(torch.load(model_path))  # carga el estado entrenado del modelo
    model.eval()  # coloca el modelo en modo evaluación
    return model

def preprocess_image(img_path):
    image = Image.open(img_path).convert('L')  # carga la imagen en escala de grises
    image = image.resize((28, 28))  # redimensiona a 28x28 píxeles
    image = np.array(image)  # convierte la imagen a un array de NumPy
    image = torch.tensor(image, dtype=torch.float32)  # convierte a tensor de PyTorch
    image = image.unsqueeze(0).unsqueeze(0)  # agrega dimensiones de lote y canal
    return image

def infer(model, image):
    with torch.no_grad():  # desactiva el cálculo de gradientes
        output = model(image)  # pasa la imagen por el modelo
        _, predicted_class = torch.max(output, 1)  # obtiene la clase predicha
    return predicted_class.item()

def main(model_path, img_path):
    model = load_model(model_path)
    image = preprocess_image(img_path)
    prediction = infer(model, image)
    print(f'La imagen fue clasificada como: {prediction}')

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("uso: python inference.py <ruta_modelo.pth> <ruta_imagen.png>")
    else:
        model_path = sys.argv[1]
        img_path = sys.argv[2]
        main(model_path, img_path)


