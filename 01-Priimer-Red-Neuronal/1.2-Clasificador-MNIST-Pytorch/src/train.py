import torch  # Importa PyTorch
import torch.optim as optim  # Importa el módulo de optimización
import torch.nn as nn  # Importa el módulo de redes neuronales

def train_model(model, train_loader, epochs=2):  # Define una función para entrenar el modelo
    criterion = nn.CrossEntropyLoss()  # Define la función de pérdida (entropía cruzada para clasificación)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Define el optimizador (descenso de gradiente estocástico)

    for epoch in range(epochs):  # Itera sobre el número de épocas
        running_loss = 0.0  # Inicializa el acumulador de pérdida
        for i, data in enumerate(train_loader, 0):  # Itera sobre el DataLoader
            inputs, labels = data  # Obtiene las entradas y etiquetas
            optimizer.zero_grad()  # Limpia los gradientes acumulados
            outputs = model(inputs)  # Realiza una pasada hacia adelante
            loss = criterion(outputs, labels)  # Calcula la pérdida
            loss.backward()  # Realiza la retropropagación del error
            optimizer.step()  # Actualiza los parámetros del modelo
            running_loss += loss.item()  # Acumula la pérdida
            if i % 1000 == 999:  # Imprime la pérdida cada 1000 batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}")
                running_loss = 0.0  # Reinicia el acumulador de pérdida
    print('Finished Training')  # Imprime un mensaje al finalizar el entrenamiento
