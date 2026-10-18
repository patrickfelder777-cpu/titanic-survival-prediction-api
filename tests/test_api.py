
from fastapi.testclient import TestClient

from serving.app import app


PASSENGER = {
    "Pclass": 1,
    "Sex": "female",
    "Age": 28,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100.0,
    "Embarked": "S",
}


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_single_prediction():
    with TestClient(app) as client:
        response = client.post("/predict", json=PASSENGER)

    assert response.status_code == 200
    result = response.json()

    assert result["survived"] in [0, 1]
    assert 0 <= result["survival_probability"] <= 1


def test_batch_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict-batch",
            json=[PASSENGER, PASSENGER],
        )

    assert response.status_code == 200
    result = response.json()

    assert len(result["predictions"]) == 2


def test_invalid_passenger():
    invalid_passenger = PASSENGER.copy()
    invalid_passenger["Age"] = "not-a-number"

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_passenger,
        )

    assert response.status_code == 422
