import joblib
import pandas as pd
import os
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
model_path = os.getenv('model_path')
artifact = joblib.load(model_path)

model = artifact["model"]
feature_columns = artifact["feature_columns"]

sample = {
    "amount": 150000,
    "hour": 2,
    "merchant_category": "Electronics",
    "location": "Chennai"
}

df = pd.DataFrame([sample])

df = pd.get_dummies(
    df,
    columns=['merchant_category', 'location'],
    drop_first=True
)

# Align columns with training
df = df.reindex(
    columns=feature_columns,
    fill_value=0
)

prediction = model.predict(df)

print("Fraud Prediction:", prediction[0])