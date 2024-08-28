# Perceptrón Multicapa (Multilayer Perceptron, MLP)

## Conceptos Clave

### 1. Introducción

El **Perceptrón Multicapa** es un tipo de red neuronal artificial que consiste en múltiples capas de neuronas, organizadas generalmente en una capa de entrada, una o más capas ocultas, y una capa de salida. Es una extensión del perceptrón simple, que permite resolver problemas no lineales al aprender representaciones complejas de los datos.

### 2. Arquitectura

- **Capas de Entrada y Salida**: 
  - La **capa de entrada** recibe las características o entradas del problema.
  - La **capa de salida** produce el resultado, que puede ser una clasificación, una predicción numérica, etc.
- **Capas Ocultas**: 
  - Entre la capa de entrada y la capa de salida se encuentran una o más capas ocultas. Estas capas permiten a la red aprender y representar patrones complejos en los datos.

  ![Diagrama MLP](images\multicapa.png)


### 3. Funcionamiento

- **Neuronas**: 
  - Cada neurona en una capa oculta o en la capa de salida calcula la suma ponderada de sus entradas (provenientes de la capa anterior), añade un sesgo (bias) y aplica una función de activación.
  
- **Pesos (Weights)**: 
  - Los **pesos** son coeficientes que se aplican a cada entrada de la neurona. Representan la importancia de cada entrada en la determinación de la salida. Durante el entrenamiento, los pesos se ajustan para minimizar el error de predicción.
  
- **Bias**:
  - El **bias** es un valor que se suma a la combinación ponderada de las entradas antes de aplicar la función de activación. Ayuda a la red a ajustarse mejor a los datos.

- **Función de Activación**: 
  - La **función de activación** introduce no linealidad en la salida de la neurona, permitiendo a la red aprender relaciones complejas entre las entradas y las salidas. Ejemplos comunes son:
    - **Sigmoide**: 
      \[
      \sigma(x) = \frac{1}{1 + e^{-x}}
      \]
    - **ReLU (Rectified Linear Unit)**:
      \[
      f(x) = \max(0, x)
      \]
    - **Tanh**:
      \[
      \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
      \]

### 4. Fórmula de Salida de la Neurona

La salida \( y \) de una neurona antes de aplicar la función de activación se calcula como:

\[
y = \sum_{i=1}^{n} (w_i \cdot x_i) + b
\]

donde:

- \( w_i \) son los pesos.
- \( x_i \) son las entradas.
- \( b \) es el bias.

La salida final \( a \) de la neurona, después de aplicar la función de activación \( \phi \), es:

\[
a = \phi \left( \sum_{i=1}^{n} (w_i \cdot x_i) + b \right)
\]

### 5. Función de Pérdida

La **función de pérdida** mide cuán lejos está la salida predicha por la red de la salida real o esperada. Es una función clave para el entrenamiento de la red, ya que el objetivo es minimizar esta pérdida ajustando los pesos de la red.

Ejemplos de funciones de pérdida:

- **Error Cuadrático Medio (MSE)** para problemas de regresión:
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]
- **Entropía Cruzada** para problemas de clasificación:
  \[
  \text{Cross-Entropy} = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
  \]

### 6. Entrenamiento y Retropropagación (backpropagation)

El entrenamiento del Perceptrón Multicapa se realiza mediante el algoritmo de **retropropagación (backpropagation)**. Este proceso involucra los siguientes pasos:

1. **Propagación hacia adelante (Forward Propagation)**:
   - Los datos de entrada se propagan a través de la red capa por capa hasta llegar a la capa de salida.

2. **Cálculo de la pérdida**:
   - Se calcula el error (pérdida) entre la salida predicha por la red y la salida real esperada.

3. **Retropropagación (Backpropagation)**:
   - El error se retropropaga a través de la red, comenzando desde la capa de salida hacia las capas anteriores. Durante este proceso, se calculan los gradientes de la función de pérdida con respecto a los pesos y biases.

4. **Actualización de Pesos (Gradient Descent)**:
   - Los pesos y biases se actualizan en la dirección opuesta al gradiente de la función de pérdida, para minimizar el error en futuras iteraciones.

Este proceso se repite durante varias iteraciones (épocas) hasta que la pérdida se minimiza y la red se ajusta a los datos de entrenamiento.

### 7. Aplicaciones

El Perceptrón Multicapa es útil en una variedad de tareas, como:

- **Clasificación**: Por ejemplo, clasificación de dígitos escritos a mano (MNIST).
- **Regresión**: Predicción de valores continuos, como precios de viviendas.
- **Reconocimiento de Patrones**: En imágenes, texto, y otros tipos de datos.

---
