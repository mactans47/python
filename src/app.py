import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_mysqldb import MySQL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'ecampus_mood481'
mysql = MySQL(app)

#solution for error 'the session is unavailable because no secret key was set'
app.secret_key = 'mysecretkey' 

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register')
def register():
    cur = mysql.connection.cursor()
    cur.execute('SELECT username, password, firstname, lastname, email FROM mdlns_user1')
    data = cur.fetchall()
    return render_template('register.html', users = data)

@app.route('/add_register', methods=['POST'])
def add_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']

        cur = mysql.connection.cursor()
        cur.execute('INSERT INTO mdlns_user1(username, password, firstname, lastname, email) VALUES(%s, %s, %s, %s, %s)', (username, password, firstname, lastname, email))
        mysql.connection.commit()

        flash('Contact added successfully')
        return redirect(url_for('register'))     


# Datos de ejemplo: años y población (en millones)
#años = np.array([2000, 2005, 2010, 2015, 2020]).reshape(-1, 1)
#población = np.array([8.0, 8.5, 9.0, 9.7, 10.2])  # Población en millones

#function for data read
def data_read_mysql():
    cur = mysql.connection.cursor()
    cur.execute("SELECT year, num_men_vot, num_wom_vot FROM voting_population WHERE department = 'Chuquisaca'")
    
    # Obtener los datos de la consulta
    resultados = cur.fetchall()
    
    # Separar los años y la población en arrays de numpy
    años = np.array([fila[0] for fila in resultados]).reshape(-1, 1)
    población = np.array([fila[1] for fila in resultados])
    población2 = np.array([fila[2] for fila in resultados])
    
    # Cerrar la conexión
    cur.close()
    #conexion.close()
    
    #return años, población, población2
    return años.flatten().tolist(), población.tolist(), población2.tolist()

# Función para crear el modelo de la red neuronal
def crear_modelo():
    modelo = Sequential()
    modelo.add(Dense(64, input_dim=1, activation='relu'))
    modelo.add(Dense(32, activation='relu'))
    modelo.add(Dense(1))
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    return modelo

# Ruta principal
@app.route('/')
def index():

    años, población, población2 = data_read_mysql()

    # Creamos y entrenamos el modelo
    modelo = crear_modelo()
    modelo.fit(años, población, epochs=1000, verbose=0)
    
    # Hacemos la predicción para el año 2025
    año_predicción = 2024
    población_predicha = modelo.predict(np.array([[año_predicción]]))
    predicción = población_predicha[0][0]    

    # Devolvemos el resultado en una plantilla HTML
    return render_template('data.html', anio=año_predicción, predicción=f"{predicción:.0f}")    

# Ruta para generar el gráfico
@app.route('/plot.png')
def plot_png():
    
    años, población, población2 = data_read_mysql()

    # Creamos y entrenamos el modelo
    modelo = crear_modelo()
    modelo.fit(años, población, epochs=1000, verbose=0)
    
    # Predicción para varios años
    años_futuros = np.array([[2012], [2013], [2024], [2018], [2024], [2025], [2026]])
    población_predicha = modelo.predict(años_futuros)

    # Graficamos los datos históricos y la predicción
    plt.figure(figsize=(8,6))
    plt.scatter(años, población, color='blue', label='Datos Históricos')
    plt.plot(años, modelo.predict(años), color='red', label='Modelo Ajustado')
    plt.scatter(años_futuros, población_predicha, color='green', marker='x', label='Predicciones Futuras')
    plt.xlabel('Año')
    plt.ylabel('Población (millones)')
    plt.title('Predicción de la Población')
    plt.legend()

    # Guardamos el gráfico en un objeto de bytes en memoria
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Enviamos la imagen al navegador
    return send_file(img, mimetype='image/png')

@app.route('/votation')
def votation():
    años, población, población2 = data_read_mysql()
    return render_template('population.html', anios=años, poblacion=población, poblacion2=población2)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)