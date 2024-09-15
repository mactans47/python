import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Cargar o entrenar el modelo previamente
# Supongamos que ya tienes el modelo entrenado (RandomForestClassifier)
# y cargado en la variable `model`

# Para este ejemplo, entrenamos uno sencillo
# Datos de ejemplo (sepal length, sepal width, petal length, petal width)
# data = {
#     'SepalLength': [5.1, 7.0, 6.3],
#     'SepalWidth': [3.5, 3.2, 3.3],
#     'PetalLength': [1.4, 4.7, 6.0],
#     'PetalWidth': [0.2, 1.4, 2.5],
#     'Species': ['setosa', 'versicolor', 'virginica']
# }

data = pd.read_csv('flores.csv')

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

data['Species'] = encoder.fit_transform(data['Species'])

df = pd.DataFrame(data)

# # Entrenar un modelo rápido
# X = df.iloc[:, :-1]  # Características (features)
# y = df.iloc[:, -1]  # Etiqueta (label)

X = data.drop(['Species'], axis=1)  # Todas las columnas excepto la última
Y = data['Species']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Título de la aplicación
st.title('Clasificación de Flores')

# Entradas del usuario para las características de la flor
sepal_length = st.number_input('Longitud del Sépalo (cm)', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input('Ancho del Sépalo (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input('Longitud del Pétalo (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input('Ancho del Pétalo (cm)', min_value=0.0, max_value=10.0, step=0.1)

# Botón para realizar la predicción
if st.button('Evaluar'):
    # Crear un array con los datos de entrada del usuario
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Realizar la predicción
    prediction = model.predict(input_data)

    # Mostrar el resultado de la predicción
    st.write(f'La flor es de la especie: {prediction[0]}')

# Para ejecutar esta aplicación:
# 1. Guarda este script en un archivo llamado `app.py`
# 2. En la terminal, ejecuta: streamlit run app.py