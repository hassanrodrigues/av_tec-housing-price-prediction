import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("dataset/Housing.csv")
df["comodos"] = df["bedrooms"] + df["bathrooms"]
df["area_comodo"] = df["area"] / df["comodos"]
df["area_comodo"] = df["area_comodo"].fillna(0)
le = LabelEncoder()
categorical_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt"],
    "max_samples": [0.8, 1.0],
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="neg_mean_squared_error",
)

try:
    grid_search.fit(X_train, y_train)

    print("Melhores parâmetros encontrados:")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nResultados do modelo:")
    print(f"Mean Squared Error: {mse:,.2f}")
    print(f"Mean Absolute Error: {mae:,.2f}")
    print(f"R² Score: {r2:.4f}")

    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": best_model.feature_importances_}
    )
    print("\nImportância das features:")
    print(feature_importance.sort_values("importance", ascending=False))

except Exception as e:
    print(f"Ocorreu um erro: {str(e)}")
