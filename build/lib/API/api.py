import joblib
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import os

class Features(BaseModel):
    bathrooms: int = Field(title="Number of bathrooms",
                           ge=1, le=6,
                           description="Number of bathrooms must be between 1 and 6")
    garage_spaces: int = Field(title="Number of garage spaces",
                               ge=1, le=3,
                               description="Number of garage spaces must be between 1 and 3")
    has_pool: bool = Field(title="Has Pool",
                           description="Must be either True or False")

app = FastAPI()
API_path = os.path.dirname(__file__)
model_path = os.path.join(API_path, 'model.joblib')
backup_path = os.path.join(API_path, './backup')
model = joblib.load(model_path)
os.makedirs(backup_path, exist_ok=True)

@app.post("/predict/")
async def predict_price(features: Features):
    try:
        input = pd.DataFrame({'bathrooms': features.bathrooms,
                              'garage_spaces': features.garage_spaces,
                              'has_pool': features.has_pool}, index=[0])
        
        predicted_price = model.predict(input)[0]
        return {"predicted_price": predicted_price}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/update-model/")
async def update_model(file: UploadFile = File()):
    try:
        global model
        new_data = pd.read_csv(file.file)
        
        required_columns = {'bathrooms', 'has_pool', 'garage_spaces', 'sale_price'}
        if not required_columns.issubset(new_data.columns):
            raise ValueError(f"Data must contain the following columns: {required_columns}")
        
        train_data_path = os.path.join(API_path, './train_data.csv')
        if os.path.exists(train_data_path):
            existing_data = pd.read_csv(train_data_path)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            updated_data = new_data
        updated_data.to_csv(train_data_path, index=False)
        

        X_train = new_data[['bathrooms', 'has_pool', 'garage_spaces']]
        y_train = new_data['sale_price']
        

        model_base = RandomForestRegressor(max_depth=1, n_estimators=50, random_state=1)
        

        categorical_transformer = OneHotEncoder(handle_unknown='ignore')


        preprocessor = ColumnTransformer(
            transformers=[('cat', categorical_transformer, ['has_pool'])]
        )


        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model_base)
        ])
        

        pipeline.fit(X_train, y_train)


        current_date = datetime.now().strftime('%Y%m%d')
        backup_filename = f'model{current_date}.joblib'
        backup_file_path = os.path.join(backup_path, backup_filename)
        joblib.dump(model, backup_file_path)
        
        model = pipeline
        joblib.dump(pipeline, model_path)
        
        return {"message": "Model updated successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
