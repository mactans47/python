import streamlit as st
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
#from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Para guardar y cargar modelos

# Título de la aplicación
st.title('Clasificación de Flores con Machine Learning')

# Sección 1: Cargar un archivo CSV
#st.header('1. Cargar un archivo CSV')
#uploaded_file = st.file_uploader("Elige un archivo CSV", type=["csv"])
uploaded_file = 'flores.csv'

def createModel(uploaded_file):    

    #if uploaded_file is not None:
    # Leer el archivo CSV
    df = pd.read_csv(uploaded_file)
    #df = pd.read_csv('flores.csv')

    #st.write("Datos cargados:")
    st.write(df.head())

    # encoder = LabelEncoder()    
    # df['Species'] = encoder.fit_transform(df['Species'])

    # Asumiendo que la última columna es la etiqueta y el resto son características
    X = df.iloc[:, :-1]  # Características
    y = df.iloc[:, -1]   # Etiqueta

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    #model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Definir el espacio de búsqueda de hiperparámetros
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [10, 20, 30, None],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'bootstrap': [True, False]
    # }

    # # Crear un GridSearchCV para encontrar los mejores hiperparámetros
    # grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X_train, y_train)

    # # Usar el mejor modelo encontrado
    # model = grid_search.best_estimator_

    # Realizar predicciones y mostrar precisión
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Precisión del modelo: {accuracy:.2f}')

    joblib.dump(model, 'model_trainer.pkl')

    data = [y_test, y_pred, df]

    print('Modelo creado con precisión de predicción: ' + str(accuracy))

    return data

def readModel():
    try:
        model = joblib.load('model_trainer.pkl')
        #st.write("Modelo cargado correctamente.")
        print("Modelo cargado correctamente.")
    except FileNotFoundError:
        #st.write("No se encontró un modelo guardado. Entrena y guarda un modelo primero.")
        print("No se encontró un modelo guardado. Entrena y guarda un modelo primero.")
    return model

data = createModel(uploaded_file)

# Sección 2: Visualizar estadísticas del modelo
st.header('Estadísticas del modelo')

#Mostrar matriz de confusión
st.subheader('Matriz de Confusión:')
cm = confusion_matrix(data[0], data[1])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

#Mostrar reporte de clasificación
st.subheader('Reporte de Clasificación:')
report = classification_report(data[0], data[1], output_dict=True)
st.write(pd.DataFrame(report).transpose())

# # Mostrar métricas adicionales
# st.subheader('Métricas adicionales')
# precision = report['weighted avg']['precision']
# recall = report['weighted avg']['recall']
# f1_score = report['weighted avg']['f1-score']
# st.write(f"Precisión (Precision): {precision:.2f}")
# st.write(f"Sensibilidad (Recall): {recall:.2f}")
# st.write(f"F1-Score: {f1_score:.2f}")
#****************************************************************************************
# Sección 3: Introducir datos manualmente
st.header('Nuevo registro de datos')
sepal_length = st.number_input('Longitud del Sépalo (cm)', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input('Ancho del Sépalo (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input('Longitud del Pétalo (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input('Ancho del Pétalo (cm)', min_value=0.0, max_value=10.0, step=0.1)

# Botón para realizar la predicción manual
if st.button('Evaluar'): 
        model = readModel()       
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
        df = pd.concat([data[2], new_data], ignore_index=True)

        # Guardar el DataFrame actualizado en el mismo archivo CSV
        #df.to_csv(uploaded_file.name, index=False)
        df.to_csv(uploaded_file, index=False)

        # Sección 3: Mostrar gráfico de líneas
        st.header('3. Gráfico de Líneas de Predicciones')
        
        # Crear un DataFrame para las predicciones de prueba
        df_test_predictions = pd.DataFrame({
            'Index': range(len(data[0])),
            'Species': data[0],
            'Type': 'Test Data'
        })

        # Crear un DataFrame para la predicción manual
        df_manual_prediction = pd.DataFrame({
            'Index': [len(data[1])],  # Agregar al final del índice
            'Species': prediction,
            'Type': 'Manual Input'
        })

         # Combinar los resultados de prueba y los datos manuales
        df_predictions = pd.concat([df_test_predictions, df_manual_prediction])

        # Crear el gráfico
        # plt.figure(figsize=(10, 5))
        # plt.plot(df2.index, df2['Species'], marker='o')
        # plt.xlabel('Índice')
        # plt.ylabel('Especie Predicha')
        # plt.title('Predicciones de Especies de Flores')
        # plt.grid(True)
        # plt.xticks(df2.index, labels=df2['Species'], rotation=45)

        #df_predictions[df_predictions['Species'] == species]['Species']
        
        species_counts = df_predictions['Species'].value_counts()

        plt.figure(figsize=(10, 5))
        for species in df_predictions['Species'].unique():
            plt.plot(df_predictions[df_predictions['Species'] == species]['Index'],
                        df_predictions[df_predictions['Species'] == species]['Species'],
                        label=species)

        plt.xlabel('Índice')
        plt.ylabel('Especie Predicha')
        plt.title('Predicciones de Especies de Flores')
        plt.legend(title='Especie')
        plt.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(plt)


        # Contar las ocurrencias de cada especie en forma acumulativa
        df_predictions['CumulativeCount'] = df_predictions.groupby('Species').cumcount() + 1

        # Crear un índice secuencial para el total de predicciones
        df_predictions['PredictionNumber'] = range(1, len(df_predictions) + 1)
        
        # Sección 3: Mostrar gráfico de líneas
        st.header('3. Gráfico de Líneas de Conteo Acumulativo de Especies Predichas')

        # Crear el gráfico
        plt.figure(figsize=(10, 6))
        for species in df_predictions['Species'].unique():
            species_data = df_predictions[df_predictions['Species'] == species]
            #plt.plot(species_data['Index'], species_data['CumulativeCount'], marker='o', label=species)
            plt.plot(species_data['PredictionNumber'], species_data['CumulativeCount'], marker='o', label=species)

        plt.xlabel('Número de Predicción')
        plt.ylabel('Conteo Acumulativo')
        plt.title('Conteo Acumulativo de Especies Predichas')
        plt.legend(title='Especie')
        plt.grid(True)

        st.write(species_data.head())

        # Mostrar el gráfico en Streamlit
        st.pyplot(plt)


        # Sección 3: Mostrar gráfico circular
        st.header('3. Gráfico Circular de Predicciones')

        # Contar las ocurrencias de cada especie predicha
        species_counts = df_predictions['Species'].value_counts()

        # Crear el gráfico circular
        plt.figure(figsize=(8, 8))
        plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribución de las Especies Predichas')
        plt.axis('equal')  # Asegura que el gráfico sea circular

        st.pyplot(plt)

        #st.write("El dataset del archivo CSV fue actualizado con este nuevo resultado.")
        st.write('Success')
    #else:
        #st.write("Por favor, carga primero un archivo CSV.")


