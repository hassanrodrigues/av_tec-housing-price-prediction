import joblib
import pandas as pd
from typing import Dict

MODEL_DIR = "model"


def load_artifacts():
    """
    Carrega os modelos e scalers previamente treinados.

    Returns:
        Tuple contendo os modelos e os scalers.
    """
    lr = joblib.load(f"{MODEL_DIR}/linear_regression_model.pkl")
    rf = joblib.load(f"{MODEL_DIR}/random_forest_model.pkl")
    scaler_X = joblib.load(f"{MODEL_DIR}/scaler_X.pkl")
    scaler_y = joblib.load(f"{MODEL_DIR}/scaler_y.pkl")
    return lr, rf, scaler_X, scaler_y


def preprocess_input(raw_input: Dict):
    """
    Pré-processa os dados de entrada conforme o pipeline de treino:
    - Codifica variáveis categóricas
    - Aplica o scaler

    Args:
        raw_input (dict): Dados da casa.

    Returns:
        Dados prontos para inferência.
    """
    df = pd.DataFrame([raw_input])

    binary_map = {"yes": 1, "no": 0}
    for col in [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
    ]:
        df[col] = df[col].map(binary_map)

    furnishing_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    df["furnishingstatus"] = df["furnishingstatus"].map(furnishing_map)

    return df


def predict_price(data: Dict, model_type: str = "rf"):
    """
    Realiza a previsão do preço com base nos dados fornecidos.

    Args:
        data (dict): Dados da casa no formato dicionário.
        model_type (str): 'lr' para regressão linear ou 'rf' para random forest.

    Returns:
        float: Preço estimado.
    """
    lr, rf, scaler_X, scaler_y = load_artifacts()

    df = preprocess_input(data)
    X_scaled = scaler_X.transform(df)

    if model_type == "lr":
        model = lr
    elif model_type == "rf":
        model = rf
    else:
        raise ValueError("Tipo de modelo inválido. Use 'lr' ou 'rf'.")

    y_scaled_pred = model.predict(X_scaled).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_scaled_pred)

    return float(y_pred[0][0])


if __name__ == "__main__":
    sample_input = {
        "area": 25,
        "bedrooms": 1,
        "bathrooms": 1,
        "stories": 1,
        "mainroad": "no",
        "guestroom": "no",
        "basement": "yes",
        "hotwaterheating": "no",
        "airconditioning": "no",
        "parking": 1,
        "prefarea": "yes",
        "furnishingstatus": "furnished",
    }

    predicted_price = predict_price(sample_input, model_type="rf")
    print(f"Preço estimado da casa: {predicted_price:,.2f}")
