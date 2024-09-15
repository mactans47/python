#importamos las librerias necesarias
from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive

X_train, y_train, X_test, y_test = titanic_survive()

# Verificar que las formas coincidan
print(X_train.shape, y_train.shape)  # Deberían tener el mismo número de filas
print(X_test.shape, y_test.shape)    # Deberían tener el mismo número de filas

model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

explainer = ClassifierExplainer(
                model, X_test, y_test,
                # optional:
                cats=['Sex', 'Deck', 'Embarked'],
                labels=['Not survived', 'Survived'])

db = ExplainerDashboard(explainer, title="Titanic Explainer",
                    whatif=False, # you can switch off tabs with bools
                    shap_interaction=False,
                    decision_trees=False)
db.run(port=8051)