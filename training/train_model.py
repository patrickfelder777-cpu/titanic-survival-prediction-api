import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import FEATURES, make_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/titanic.csv')
    parser.add_argument('--output', default='models')
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df['Survived'], test_size=0.2, stratify=df['Survived'], random_state=42)
    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
    metrics = {'accuracy': float(accuracy_score(y_test, prediction)),
               'f1': float(f1_score(y_test, prediction)),
               'roc_auc': float(roc_auc_score(y_test, probability)),
               'confusion_matrix': confusion_matrix(y_test, prediction).tolist()}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output / 'titanic_pipeline.joblib')
    metadata = {'model_type': 'RandomForestClassifier', 'features': FEATURES,
                'training_date': datetime.now(timezone.utc).isoformat(),
                'train_rows': len(X_train), 'test_rows': len(X_test),
                'metrics': metrics, 'pipeline_version': 2}
    (output / 'model_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
