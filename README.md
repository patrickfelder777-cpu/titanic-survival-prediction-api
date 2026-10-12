# Titanic Survival Prediction API

Random Forest classification with a FastAPI service and separate Docker definitions for training and serving. Based on Patrick Felder's Titanic coursework project; this portfolio revision fixes preprocessing leakage and training/serving differences.

## Architecture
A saved scikit-learn Pipeline contains deterministic family-size features, numeric median imputation, categorical mode imputation and one-hot encoding, and a Random Forest classifier. Raw rows are split 80/20 with stratification and random_state=42 before preprocessing is fitted. The API uses this exact saved pipeline.

The revised model uses seven passenger inputs plus FamilySize and IsAlone. It removes the original Title, AgeGroup, and FareGroup features to avoid the original guessed titles and inconsistent binning. This is a changed model and requires retraining; old model binaries are incompatible.

## Local setup
Use Python 3.11 in a virtual environment, from the repository root:

```bash
pip install -r requirements.txt
python training/train_model.py --data data/titanic.csv
uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

Supply your authorized Titanic CSV containing Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, and Survived. The original source notebook reports 891 rows. Data is excluded from this repository. The training command creates models/titanic_pipeline.joblib and models/model_metadata.json. Only load trusted joblib files.

Interactive API documentation: http://localhost:8000/docs. Use sample_input.json for POST /predict.

## Docker
Run these commands from the repository root after creating data/ and models/. Absolute host paths are required for the mounts; replace the examples with your own paths.

```bash
docker build -f training/Dockerfile -t titanic-training .
docker run --rm -v /absolute/path/data:/app/data:ro -v /absolute/path/models:/app/models titanic-training
docker build -f serving/Dockerfile -t titanic-api .
docker run --rm -p 127.0.0.1:8000:8000 -v /absolute/path/models:/app/models:ro titanic-api
```

PowerShell users can use quoted Windows host paths instead of the example paths. Both containers share the same requirements and pipeline module. No Docker Hub publication or public deployment is claimed for this revision.

## Endpoints
- GET /: application status
- GET /health: readiness; returns 503 without a loaded model
- GET /model-info: saved evaluation metadata
- POST /predict: survival class and model probability
- POST /predict-batch: 1–1,000 passenger requests

Inputs validate passenger class, sex, embarkation port, nonnegative counts/fare, and finite age between 0 and 120. Model probability is not calibrated confidence or a real-world safety forecast.

## Evaluation and provenance
The supplied original metadata reports accuracy **0.8156424581 (81.56%)**. That result belongs to the historical implementation, which fitted imputation/encoding and fare bins before the split and used different preprocessing during serving. It is preserved in historical/ for transparency and must not be presented as a score for this revision.

The corrected training script reports accuracy, F1, ROC-AUC, and a confusion matrix on its held-out split. It has not been trained on the real Titanic dataset here because the CSV was not supplied. Python syntax and eight API integration checks passed with a synthetic fixture, covering predictions, batches, readiness, and invalid inputs. Synthetic checks establish integration behavior, not Titanic model quality. Docker builds were not executed. Dependency pins are intended for Python 3.11 and the container environment was not built during preparation.

This is a portfolio demonstration, not a production service. Public hosting would require further operational work such as authentication and rate limits.
