from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from predict import predict_price
import uvicorn
app = FastAPI(title="API de Previsão de Preço de Casas", version="1.0.0")

class HouseData(BaseModel):
    area: float = 50
    bedrooms: int = 1
    bathrooms: int = 1
    stories: int = 1
    mainroad: Literal["yes", "no"] = "no"
    guestroom: Literal["yes", "no"] = "no"
    basement: Literal["yes", "no"] = "no"
    hotwaterheating: Literal["yes", "no"] = "no"
    airconditioning: Literal["yes", "no"] = "no"
    parking: int = 1
    prefarea: Literal["yes", "no"]
    furnishingstatus: Literal["unfurnished", "semi-furnished", "furnished"]
    model_type: Literal["lr", "rf"] = "rf"

@app.post("/predict")
def predict(data: HouseData):
    try:
        prediction = predict_price(data.model_dump(exclude={"model_type"}), model_type=data.model_type)
        return {
            "preco_estimado": round(prediction, 2),
            "modelo_usado": data.model_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
