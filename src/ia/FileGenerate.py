import pandas as pd
import numpy as np

# Fijar una semilla para reproducibilidad
np.random.seed(42)

# Definir el número de registros
num_records = 50

# Generar datos aleatorios para las características
sepal_length = np.random.normal(5.8, 0.8, num_records)  # Media=5.8, SD=0.8
sepal_width = np.random.normal(3.0, 0.4, num_records)   # Media=3.0, SD=0.4
petal_length = np.random.normal(3.7, 1.5, num_records)  # Media=3.7, SD=1.5
petal_width = np.random.normal(1.2, 0.7, num_records)   # Media=1.2, SD=0.7

# Generar datos categóricos para la especie
species = np.random.choice(['setosa', 'versicolor', 'virginica'], num_records)

# Crear un DataFrame
data = pd.DataFrame({
    'SepalLength': sepal_length,
    'SepalWidth': sepal_width,
    'PetalLength': petal_length,
    'PetalWidth': petal_width,
    'Species': species
})

# Guardar el DataFrame como un archivo CSV
file_path = 'flores.csv'
data.to_csv(file_path, index=False)

print(f"Archivo CSV guardado en: {file_path}")
