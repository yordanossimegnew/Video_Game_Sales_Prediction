# import the necessary libraries
from fastapi import FastAPI
import pandas as pd
import config
from preprocessor import Pipeline
from pydantic import BaseModel
from typing import List

# initializing the app
app = FastAPI()


# Initializing the pipeline
pipeline = Pipeline(config.TARGET,
                    config.LEAKY_FEATURES,
                    config.DATA_TYPE_CONVERSION,
                    config.HIGH_LOW_CARDINALITY_FEATURES,
                    config.CAT_IMP_MODE,
                    config.CAT_IMP_MISSING,
                    config.CAT_VARS,
                    config.TEMPORAL_VARIABLES,
                    config.CONT_VARS)

# Load and Fit the Data with the existing model
data = pd.read_csv(config.DATA_PATH)
pipeline.fit(data)


class InputData(BaseModel):
    Name: str
    Platform: str
    Year_of_Release: float
    Genre: str
    Publisher: str
    NA_Sales: float
    EU_Sales: float
    JP_Sales: float
    Other_Sales: float
    Critic_Score: str
    Critic_Count: float
    User_Score: float
    User_Count: float
    Developer: str
    Rating: str
    
class InputDataList(BaseModel):
    data: List[InputData]
    
@app.get("/")
def read_root():
    return {"message" : "Video Game Sales Prediction" }

@app.post("/predict")
def predict(input_data: InputDataList):
    input_df = pd.DataFrame([item.dict() for item in input_data.data])
    predictions = pipeline.predict(input_df)
    prediction_value = predictions.tolist()[0]
    message = f"The Saling Price of the Video Game is {prediction_value}"
    return {"prediction": prediction_value, "message": message}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    