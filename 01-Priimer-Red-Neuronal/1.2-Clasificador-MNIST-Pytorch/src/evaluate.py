import torch  # Importa PyTorch

def evaluate_model(model, test_loader):  # Define una función para evaluar el modelo
    correct = 0  # Inicializa el contador de predicciones correctas
    total = 0  # Inicializa el contador de ejemplos totales
    with torch.no_grad():  # Desactiva el cálculo de gradientes (ahorra memoria)
        for images, labels in test_loader:  # Itera sobre el DataLoader de prueba
            outputs = model(images)  # Realiza una pasada hacia adelante
            _, predicted = torch.max(outputs, 1)  # Obtiene las predicciones (índice de la clase con la mayor probabilidad)
            total += labels.size(0)  # Acumula el número total de etiquetas
            correct += (predicted == labels).sum().item()  # Acumula el número de predicciones correctas
    
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')  # Imprime la precisión del modelo
