from explainerdashboard.datasets import titanic_survive

# Cargar los datos de la función titanic_survive
data = titanic_survive()

# Si data es una tupla, desempaquetarla para obtener el DataFrame
if isinstance(data, tuple):
    df = data[0]  # Supongamos que el primer elemento de la tupla es el DataFrame
else:
    df = data

# Ahora, guardar el DataFrame en un archivo CSV
df.to_csv("titanic_survive.csv", index=False)

# Mostrar la estructura de los datos
#print(df.head(10))