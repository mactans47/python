#importamos las librerias necesarias
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
#from explainerdashboard.datasets import titanic_survive

import pandas as pd

#leemos el dataset
data = pd.read_csv('flores.csv')

#X = data.drop(['precio','nombre'], axis=1)
#Y = data['precio']

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
#convertimos las columnas 'unidad_de_medida','tipo_de_contratacion','modalidad','objeto_de_contratacion','departamento'
data['Species'] = encoder.fit_transform(data['Species'])

# Suponemos que la última columna es la etiqueta y el resto son características
# X = data.iloc[:, :-1]  # Todas las columnas excepto la última
# Y = data.iloc[:, -1]   # Última columna

X = data.drop(['Species'], axis=1)  # Todas las columnas excepto la última
Y = data['Species']

from sklearn.model_selection import train_test_split
#X_train, y_train, X_test, y_test = titanic_survive()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Verificar que las formas coincidan
print(X_train.shape, y_train.shape)  # Deberían tener el mismo número de filas
print(X_test.shape, y_test.shape)    # Deberían tener el mismo número de filas

#model = RandomForestClassifier(n_estimators=50, max_depth=5)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

descripcion = {
    "SepalLength": "Longitud del sépalo (cm)",
    "SepalWidth": "Ancho del sépalo (cm)",
    "PetalLength": "Longitud del pétalo (cm)",    
    "PetalWidth": "Ancho del pétalo (cm)"    
}

explainer = ClassifierExplainer(
                model, X_test, y_test,
                # optional:
                #cats=['Medida1','Medida2','Medida3','Medida4'],
                descriptions = descripcion, 
                target = 'Especie de planta',
                labels=['setosa', 'versicolor', 'virginica'])

#explainer = ClassifierExplainer(model, X_test, y_test)

db = ExplainerDashboard(explainer, title="Flowers Explainer",
                    whatif=False, # you can switch off tabs with bools
                    shap_interaction=False,
                    decision_trees=False)
db.run(debug=True, port=8051)