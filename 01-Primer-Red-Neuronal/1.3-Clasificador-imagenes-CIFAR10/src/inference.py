import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# función para cargar el modelo guardado
def load_trained_model(model_path='cifar10_model.h5'):
    return load_model(model_path)

# función para cargar y preprocesar la imagen
def preprocess_image(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # agregar una dimensión extra para el batch
    img_array = img_array / 255.0  # normalizar la imagen
    return img_array

# función para hacer la predicción y mostrar la imagen con la categoría
def predict_and_display(model, img_path, class_names):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # mostrar la imagen con la categoría predicha
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f'Predicted: {class_names[predicted_class]}')
    plt.axis('off')
    plt.show()

    return class_names[predicted_class]
