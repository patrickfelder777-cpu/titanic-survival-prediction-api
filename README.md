
# Titanic Survival Prediction API

[![Titanic API CI](https://github.com/patrickfelder777-cpu/titanic-survival-prediction-api/actions/workflows/ci.yml/badge.svg)](https://github.com/patrickfelder777-cpu/titanic-survival-prediction-api/actions/workflows/ci.yml)

An end-to-end machine learning application that predicts Titanic passenger survival using a Random Forest classifier, FastAPI REST endpoints, automated testing, GitHub Actions CI, and Docker.

This portfolio project demonstrates Python backend development, machine learning integration, reproducible preprocessing, API validation, and containerization.

## Tech Stack

- Python 3.11
- FastAPI, Pydantic, Uvicorn
- scikit-learn, pandas, NumPy
- pytest, HTTPX
- Docker
- GitHub Actions
- joblib

## Architecture

The application uses a saved scikit-learn Pipeline containing:

1. Feature engineering for FamilySize and IsAlone.
2. Numeric median imputation.
3. Categorical mode imputation.
4. One-hot encoding.
5. Random Forest classification.

The dataset is split into training and testing sets before preprocessing is fitted, preventing leakage from the held-out test set.

The FastAPI application loads the same trained pipeline used during evaluation, ensuring consistent preprocessing between training and inference.

This revision replaces preprocessing from the original coursework implementation that included inconsistent feature transformations.

## Model Performance

The revised Random Forest pipeline was trained locally using the Kaggle Titanic training dataset containing 891 passenger records.

An 80/20 stratified train/test split produced 179 held-out test samples.

| Metric | Result |
|---|---|
| Accuracy | **81.56%** |
| F1 Score | **0.736** |
| ROC-AUC | **0.835** |
| Test Samples | **179** |

### Confusion Matrix

| | Predicted Not Survived | Predicted Survived |
|---|---:|---:|
| Actual Not Survived | 100 | 10 |
| Actual Survived | 23 | 46 |

These metrics were measured on the held-out test split, not on an independent external dataset.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Application status |
| GET | `/health` | Model readiness |
| GET | `/model-info` | Model evaluation metadata |
| POST | `/predict` | Single passenger survival prediction |
| POST | `/predict-batch` | Batch predictions for 1–1,000 passengers |

The API validates passenger class, sex, embarkation port, passenger counts, fare, and age.

### Example Request

POST `/predict`

```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 28,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 100.0,
  "Embarked": "S"
}
```

### Example Response

```json
{
  "survived": 1,
  "survival_probability": 0.982907029478458
}
```

This response was verified against the locally trained model and the Dockerized serving application.

The probability is a model estimate, not calibrated confidence.

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/patrickfelder777-cpu/titanic-survival-prediction-api.git
cd titanic-survival-prediction-api
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the dataset

Download `train.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data).

Save it as:

`data/titanic.csv`

The dataset is excluded from the repository.

### 5. Train the model

```bash
python -m training.train_model --data data/titanic.csv
```

Training produces:

- `models/titanic_pipeline.joblib`
- `models/model_metadata.json`

Only load trusted joblib artifacts.

### 6. Start the API

```bash
python -m uvicorn serving.app:app --reload
```

Open the interactive Swagger documentation:

http://127.0.0.1:8000/docs

## Automated Testing

Four API integration tests were implemented using pytest and FastAPI's TestClient.

Tests cover:

- Health endpoint readiness.
- Single passenger prediction.
- Batch prediction.
- Invalid input rejection.

After training the model, run:

```bash
python -m pip install pytest httpx
python -m pytest tests/test_api.py -v
```

**Verified local result: 4 tests passed.**

## Continuous Integration

GitHub Actions automatically runs the CI workflow on pushes and pull requests.

The workflow:

1. Checks out the repository.
2. Sets up Python 3.11.
3. Installs project and testing dependencies.
4. Downloads a publicly hosted Titanic dataset.
5. Trains the Random Forest model.
6. Executes the automated API tests.

**GitHub Actions CI: Passed — verified October 18, 2026.**

The CI workflow uses a publicly hosted Titanic dataset. The performance metrics documented above come from the local Kaggle training dataset.

## Docker

The project includes separate Dockerfiles for model training and API serving:

- `training/Dockerfile`
- `serving/Dockerfile`

Both Docker images were successfully built and tested locally using Docker Desktop.

### Build the serving image

From the repository root:

```bash
docker build -f serving/Dockerfile -t titanic-api .
```

### Run the serving container

Windows PowerShell:

```powershell
docker run --rm --name titanic-api-test -p 127.0.0.1:8001:8000 -v "${PWD}\models:/app/models:ro" titanic-api
```

This command exposes the containerized API on port 8001.

API documentation:

http://127.0.0.1:8001/docs

### Build the training image

```bash
docker build -f training/Dockerfile -t titanic-training .
```

### Run training inside Docker

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path .\docker-models

docker run --rm -v "${PWD}\data:/app/data:ro" -v "${PWD}\docker-models:/app/models" titanic-training
```

The container reads the Titanic dataset and saves trained model artifacts to the `docker-models` directory.

### Docker Verification Results

Both Docker workflows were successfully validated locally.

| Component | Verified Result |
|---|---|
| Serving Docker image build | Passed |
| Training Docker image build | Passed |
| Containerized API startup | Passed |
| Docker `/health` endpoint | HTTP 200 |
| Docker `/predict` endpoint | Passed |
| Docker model training | Passed |
| Model artifact generation | Passed |
| Metadata artifact generation | Passed |

The Dockerized API returned a predicted survival probability of **98.29%** for the documented example passenger.

The Docker training container successfully generated:

- `docker-models/titanic_pipeline.joblib`
- `docker-models/model_metadata.json`

Both artifacts were verified on the host computer.

Generated model artifacts and datasets are excluded from Git tracking.

No public cloud deployment or Docker Hub publication is claimed.

## Improvements Over the Original Coursework

This revision improves the original Titanic machine learning project through:

- Leakage-aware train/test preprocessing.
- A reusable scikit-learn Pipeline.
- Consistent training and inference transformations.
- Feature engineering for family relationships.
- Persisted model artifacts and evaluation metadata.
- FastAPI REST endpoints with Pydantic validation.
- Single and batch prediction support.
- Automated API integration testing.
- GitHub Actions continuous integration.
- Separate Docker training and serving environments.

The historical implementation is retained separately for provenance.

## Limitations and Future Improvements

This is a portfolio demonstration, not a publicly hosted production service.

Potential improvements include:

- Authentication and rate limiting.
- Model monitoring and drift detection.
- Automated cloud deployment.
- Model versioning and artifact management.
- Additional testing and external validation.
- Probability calibration.

The model's predictions are educational estimates based on historical passenger data.

## Author

**Patrick Felder**

GitHub: https://github.com/patrickfelder777-cpu
