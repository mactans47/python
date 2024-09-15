import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Datos de ejemplo: años y población (en millones)
# Puedes reemplazar estos datos con tus propios datos de años y población
años = np.array([2000, 2005, 2010, 2015, 2020]).reshape(-1, 1)
población = np.array([8.0, 8.5, 9.0, 9.7, 10.2])  # Población en millones

# Creación del modelo de red neuronal
modelo = Sequential()
# Añadimos capas densas a la red neuronal
modelo.add(Dense(64, input_dim=1, activation='relu'))  # Capa oculta con 64 neuronas
modelo.add(Dense(32, activation='relu'))               # Capa oculta con 32 neuronas
modelo.add(Dense(1))                                   # Capa de salida

# Compilamos el modelo con el optimizador Adam y la función de pérdida Mean Squared Error
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamos el modelo con los datos de años y población
# epochs = número de veces que pasamos los datos por la red neuronal
modelo.fit(años, población, epochs=1000, verbose=0)

# Hacemos predicciones para años futuros
año_predicción = np.array([[2025]])  # Año para el que queremos predecir la población
población_predicha = modelo.predict(año_predicción)

# Mostramos el resultado
print(f"La población predicha para el año {año_predicción[0][0]} es de {población_predicha[0][0]:.2f} millones.")

# Opcional: visualización de los datos y la predicción
plt.scatter(años, población, color='blue', label='Datos de Población')
plt.plot(años, modelo.predict(años), color='red', label='Predicción de la Red Neuronal')

# Añadimos la predicción al gráfico
plt.scatter(año_predicción, población_predicha, color='green', marker='x', label=f'Predicción {año_predicción[0][0]}')

plt.xlabel('Año')
plt.ylabel('Población (millones)')
plt.title('Predicción de la Población usando Redes Neuronales')
plt.legend()
plt.grid(True)
plt.show()
