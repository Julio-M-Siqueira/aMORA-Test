# Real Estate Pricing Model API

## Project Overview
This project is a real estate pricing model API developed using FastAPI. It predicts house prices based on input features such as number of bathrooms, garage spaces, and whether the property has a pool.

## Notes

Additional notes and model development details are documented in `notebook.ipynb`.

## Requirements
- Python 3.11 or higher
- pip (Python package manager)
## Directory Structure

```
aMora/
|-- API
|-- |-- backup
|-- |-- api.py
|-- |-- model.joblib
|-- |-- train_data.csv
|-- |-- __init__.py
|-- Data
|-- |-- properties.csv
|-- |-- transactions.csv
|-- |-- economics.csv
|-- tests
|-- |-- test_data.csv
|-- |-- test_api.py
|-- |-- __init__.py
|-- notebook.ipynb
|-- requirements.txt
|-- setup.py
|-- tox.ini
|-- CHANGELOG.md
|-- README.md
|-- __init__.py
```

## Setup Instructions
1. Clone the repository
```bash
git clone <repository_link>
cd <repository_folder>
```

2. Set up the virtual environment
```bash
python -m venv .venv
source .venv/Scripts/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the API
1. Start the FastAPI server:
```bash
uvicorn API.api:app --reload
```

2. Access the API documentation:
```bash
http://127.0.0.1:8000/docs
```

## API Endpoints
- **`/predict/`**: POST request to predict house prices based on input features.
- **Input**: JSON with fields `bathrooms` (int), `garage_spaces` (int), and `has_pool` (bool).
- **Response**: JSON with field `predicted_price` (float).

- **`/update-model/`**: POST request to update the model with new training data.
- **Input**: CSV file with columns `bathrooms`, `garage_spaces`, `has_pool`, and `sale_price`.
- **Response**: JSON with message `"Model updated successfully"`.

## Retraining the Model
- Upload a CSV file with the required columns (`bathrooms`, `garage_spaces`, `has_pool`, `sale_price`) to the `/update-model/` endpoint. In the `Data` folder you can use the `test_data.csv` as an example of CSV file to upload.
- The updated model will be saved, and the existing model will be backed up in the `backup` folder.
## Example Usage
1. Start the API server using `uvicorn`.
2. Use API documentation for test. In your browser access:
```
http://127.0.0.1:8000/docs
```

**Or**:
1. Start the API server using `uvicorn`.
2. Run the following command for testing the `/predict/` endpoint:
```bash
curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"bathrooms": 3, "garage_spaces": 2, "has_pool": true}
```
4. Run the following command for testing the `/update-model/` endpoint to upload new data and retrain the model.
```bash
echo -e "bathrooms,garage_spaces,has_pool,sale_price\n2,1,True,300000\n3,2,False,400000" | \
curl -X POST "http://127.0.0.1:8000/update-model/" -F "file=@-"
```

## Dataset Description

- **properties.csv**: Property listings with features.
- **transactions.csv**: Historical transaction data.
- **economics.csv**: Economic indicators over time.
- **test_data.csv**: Update data for testing API.

## Testing the API
### Run tests with pytest
```bash
pytest tests/
```

### Testing individual endpoints with curl
First, start the API server using `uvicorn`.
#### Predict House Price
```bash
curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"bathrooms": 3, "garage_spaces": 2, "has_pool": true}
```
#### Update the model with simulated data
```bash
echo -e "bathrooms,garage_spaces,has_pool,sale_price\n2,1,True,300000\n3,2,False,400000" | \
curl -X POST "http://127.0.0.1:8000/update-model/" -F "file=@-"
```

## Author

Developed by Julio.