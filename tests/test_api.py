import pytest
from fastapi.testclient import TestClient
from ..API.api import app

client = TestClient(app)

def test_predict_price_success():
    """Tests the /predict/ endpoint with valid input.

    Args:
        None

    Returns:
        bool: True if the test passes (response status is 200 
        and contains 'predicted_price'), otherwise False.

    Raises:
        AssertionError: If the expected conditions are not met.

    Notes:
        See https://fastapi.tiangolo.com/tutorial/testing/ for more info
    """
    response = client.post(
        "/predict/",
        json={
            "bathrooms": 3,
            "garage_spaces": 2,
            "has_pool": True
        }
    )
    assert response.status_code == 200
    assert "predicted_price" in response.json()

def test_predict_price_invalid_bathrooms():
    """Tests the /predict/ endpoint with invalid 'bathrooms' input.

    Args:
        None

    Returns:
        bool: True if the test passes (response status is 422), otherwise False.

    Raises:
        AssertionError: If the expected conditions are not met.

    Notes:
        See https://fastapi.tiangolo.com/tutorial/testing/ for more info
    """
    response = client.post(
        "/predict/",
        json={
            "bathrooms": 7,  # Invalid, as bathrooms must be between 1 and 6
            "garage_spaces": 2,
            "has_pool": True
        }
    )
    assert response.status_code == 422

def test_update_model_missing_columns():
    """Tests the /update-model/ endpoint with missing columns in the uploaded CSV data.

    Args:
        None

    Returns:
        bool: True if the test passes (response status is 500 and contains 
        the expected error message), otherwise False.

    Raises:
        AssertionError: If the expected conditions are not met.

    Notes:
        See https://fastapi.tiangolo.com/tutorial/testing/ for more info
    """
    # Create an invalid CSV data
    invalid_csv_data = "bathrooms,garage_spaces,has_pool\n2,1,True\n"
    
    response = client.post(
        "/update-model/",
        files={"file": ("test.csv", invalid_csv_data)}
    )
    assert response.status_code == 500
    assert "Data must contain the following columns" in response.json()["detail"]

def test_predict_price_invalid_garage_spaces():
    """Tests the /predict/ endpoint with invalid 'garage_spaces' input.

    Args:
        None

    Returns:
        bool: True if the test passes (response status is 422), otherwise False.

    Raises:
        AssertionError: If the expected conditions are not met.

    Notes:
        See https://fastapi.tiangolo.com/tutorial/testing/ for more info
    """
    response = client.post(
        "/predict/",
        json={
            "bathrooms": 3,
            "garage_spaces": 5,  # Invalid, must be between 1 and 3
            "has_pool": False
        }
    )
    assert response.status_code == 422

def test_update_model_success():
    """Tests the /update-model/ endpoint with valid CSV data.

    Args:
        None

    Returns:
        bool: True if the test passes (response status is 200 and success 
        message is received), otherwise False.

    Raises:
        AssertionError: If the expected conditions are not met.

    Notes:
        See https://fastapi.tiangolo.com/tutorial/testing/ for more info
    """
    valid_csv_data = "bathrooms,garage_spaces,has_pool,sale_price\n2,1,True,300000\n"
    
    response = client.post(
        "/update-model/",
        files={"file": ("test.csv", valid_csv_data)}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Model updated successfully"
