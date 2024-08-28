# model.py

import tensorflow as tf
from tensorflow.keras import layers, models

# carga y normaliza los datos
def load_and_normalize_data():
    # carga el conjunto de datos CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # normaliza los datos dividiendo por 255 para que estén en el rango [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

# construye un modelo de red neuronal convolucional
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# compila el modelo con el optimizador, la función de pérdida y las métricas
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# entrena el modelo con los datos de entrenamiento y validación
def train_model(model, x_train, y_train, x_test, y_test):
    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_test, y_test))
    return history

# guarda el modelo entrenado en un archivo
def save_model(model, filename='cifar10_model.h5'):
    model.save(filename)

# grafica el desempeño del modelo durante el entrenamiento
def plot_performance(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
