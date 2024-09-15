import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Título de la aplicación
st.title('Clasificación de Flores con Machine Learning')

# Sección 1: Cargar un archivo CSV
st.header('1. Cargar un archivo CSV')
#uploaded_file = st.file_uploader("Elige un archivo CSV", type=["csv"])
uploaded_file = 'flores.csv'

#if uploaded_file is not None:
    # Leer el archivo CSV
df = pd.read_csv(uploaded_file)
#df = pd.read_csv('flores.csv')

st.write("Datos cargados:")
st.write(df.head())

# Asumiendo que la última columna es la etiqueta y el resto son características
X = df.iloc[:, :-1]  # Características
y = df.iloc[:, -1]   # Etiqueta

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones y mostrar precisión
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Precisión del modelo: {accuracy:.2f}')

# Sección 2: Visualizar estadísticas del modelo
st.header('2. Visualizar estadísticas del modelo')

# Mostrar matriz de confusión
st.subheader('Matriz de Confusión')
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Mostrar reporte de clasificación
st.subheader('Reporte de Clasificación')
report = classification_report(y_test, y_pred, output_dict=True)
st.write(pd.DataFrame(report).transpose())

# Mostrar métricas adicionales
st.subheader('Métricas adicionales')
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']
st.write(f"Precisión (Precision): {precision:.2f}")
st.write(f"Sensibilidad (Recall): {recall:.2f}")
st.write(f"F1-Score: {f1_score:.2f}")
#****************************************************************************************
# Sección 3: Introducir datos manualmente
st.header('3. Introducir datos manualmente')
sepal_length = st.number_input('Longitud del Sépalo (cm)', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input('Ancho del Sépalo (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input('Longitud del Pétalo (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input('Ancho del Pétalo (cm)', min_value=0.0, max_value=10.0, step=0.1)

# Botón para realizar la predicción manual
if st.button('Evaluar'):
    #if uploaded_file is not None:  # Verificar que los datos han sido cargados
        # Crear un array con los datos de entrada del usuario
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Realizar la predicción
        prediction = model.predict(input_data)

        # Mostrar el resultado de la predicción
        st.write(f'La flor es de la especie: {prediction[0]}')
        
        # Crear un nuevo DataFrame con los datos introducidos y el resultado
        new_data = pd.DataFrame({
            'SepalLength': [sepal_length],
            'SepalWidth': [sepal_width],
            'PetalLength': [petal_length],
            'PetalWidth': [petal_width],
            'Species': [prediction[0]]
        })

        # Concatenar el nuevo registro con los datos existentes
        df = pd.concat([df, new_data], ignore_index=True)

        # Guardar el DataFrame actualizado en el mismo archivo CSV
        #df.to_csv(uploaded_file.name, index=False)
        df.to_csv(uploaded_file, index=False)

        st.write("El dataset del archivo CSV fue actualizado con este nuevo resultado.")
    #else:
        #st.write("Por favor, carga primero un archivo CSV.")
    
