import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def load_and_preprocess_data(filepath: str):
    """
    Carrega o dataset e aplica pré-processamento básico:
    - Codificação de variáveis categóricas
    - Separação entre features e target

    Args:
        filepath (str): Caminho para o arquivo CSV.

    Returns:
        Tuple contendo os DataFrames X (features) e y (target).
    """
    df = pd.read_csv(filepath)

    # Codifica colunas categóricas binárias como 0/1
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Codifica coluna 'furnishingstatus' com inteiros
    df['furnishingstatus'] = LabelEncoder().fit_transform(df['furnishingstatus'])

    # Separa variáveis independentes (X) da variável alvo (y)
    X = df.drop("price", axis=1)
    y = df["price"]

    return X, y


def scale_data(X, y):
    """
    Normaliza os dados utilizando Min-Max Scaling.

    Args:
        X (DataFrame): Features de entrada.
        y (Series): Target.

    Returns:
        Tuple com arrays escalados e instâncias dos scalers treinados.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    return X_scaled, y_scaled, scaler_X, scaler_y


def train_models(X_train, y_train):
    """
    Instancia e treina os modelos de Regressão Linear e Random Forest.

    Args:
        X_train (array): Dados de treino para features.
        y_train (array): Dados de treino para target.

    Returns:
        Tuple com os modelos treinados.
    """
    lr = LinearRegression()
    rf = RandomForestRegressor(random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return lr, rf


def evaluate_model(model, X_test, y_test):
    """
    Realiza a avaliação de um modelo utilizando métricas padrão de regressão.

    Args:
        model: Instância do modelo treinado.
        X_test (array): Features de teste.
        y_test (array): Target de teste.

    Returns:
        Tuple com MSE, MAE e R² score.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2


def save_models_and_scalers(lr, rf, scaler_X, scaler_y, output_dir="model"):
    """
    Salva os modelos treinados e os scalers em disco utilizando joblib.

    Args:
        lr: Modelo de regressão linear.
        rf: Modelo Random Forest.
        scaler_X: Scaler utilizado para as features.
        scaler_y: Scaler utilizado para o target.
        output_dir (str): Caminho do diretório de saída.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(lr, os.path.join(output_dir, "linear_regression_model.pkl"))
    joblib.dump(rf, os.path.join(output_dir, "random_forest_model.pkl"))
    joblib.dump(scaler_X, os.path.join(output_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(output_dir, "scaler_y.pkl"))


def main():
    """
    Pipeline principal de treinamento:
    - Carrega e prepara os dados
    - Normaliza as variáveis
    - Divide dataset em treino/teste
    - Treina os modelos
    - Avalia o desempenho
    - Persiste os modelos e transformadores
    """
    X, y = load_and_preprocess_data("dataset/Housing.csv")
    X_scaled, y_scaled, scaler_X, scaler_y = scale_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    lr, rf = train_models(X_train, y_train)

    print("Regressão Linear:")
    lr_metrics = evaluate_model(lr, X_test, y_test)
    print(f"MSE: {lr_metrics[0]:.5f}, MAE: {lr_metrics[1]:.5f}, R²: {lr_metrics[2]:.5f}")

    print("\nRandom Forest:")
    rf_metrics = evaluate_model(rf, X_test, y_test)
    print(f"MSE: {rf_metrics[0]:.5f}, MAE: {rf_metrics[1]:.5f}, R²: {rf_metrics[2]:.5f}")

    save_models_and_scalers(lr, rf, scaler_X, scaler_y)


if __name__ == "__main__":
    main()
