import pandas as pd
import mysql.connector

# Conexión a la base de datos MySQL
def conectar_bd():
    return mysql.connector.connect(
        host="localhost",  # Cambia por la IP o nombre de tu servidor MySQL
        user="root",  # Cambia por tu nombre de usuario
        password="",  # Cambia por tu contraseña
        database="ecampus_mood481"  # Cambia por el nombre de tu base de datos
    )

# Leer CSV
def cargar_csv_a_bd(archivo_csv, tabla_destino):
    # Leer archivo CSV en un DataFrame de pandas
    df = pd.read_csv(archivo_csv)
    
    # Conexión a la base de datos
    conexion = conectar_bd()
    cursor = conexion.cursor()

    # Crear consulta de inserción
    columnas = ', '.join(df.columns)  # Obtener nombres de las columnas
    valores = ', '.join(['%s'] * len(df.columns))  # Crear placeholders para valores
    consulta_sql = f"INSERT INTO {tabla_destino} ({columnas}) VALUES ({valores})"

    # Insertar cada fila del DataFrame en la tabla MySQL
    for fila in df.itertuples(index=False, name=None):
        cursor.execute(consulta_sql, fila)

    # Confirmar cambios y cerrar conexión
    conexion.commit()
    cursor.close()
    conexion.close()
    print(f"Datos insertados correctamente en la tabla {tabla_destino}")

# Ejecutar el script
if __name__ == "__main__":
    archivo_csv = "pob_votantes.csv"  # Cambia por la ruta de tu archivo CSV
    tabla_destino = "voting_population"  # Cambia por el nombre de la tabla destino en MySQL
    cargar_csv_a_bd(archivo_csv, tabla_destino)
