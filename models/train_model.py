"""Training pipeline for churn prediction using classical ML models and SMOTE balancing."""

import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
import json

print("Loading data...")
df = pd.read_csv('cleaned_data.csv')

# cleaning script one-hot encoded Churn, we just want a single binary column back
df['Churn'] = df['Churn_1']
df.drop(['Churn_0', 'Churn_1'], axis=1, inplace=True)

print(f"Dataset shape: {df.shape}")

# convert target to int just in case it came in as a string
if df['Churn'].dtype == object:
    df['Churn'] = LabelEncoder().fit_transform(df['Churn'])

X = df.drop('Churn', axis=1)
y = df['Churn']
feature_names = X.columns.tolist()

# split FIRST before any encoding — otherwise we're leaking test data into training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# fit encoders on train only, then apply to test — same rule, no leakage
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    # unseen categories in test get mapped to -1 instead of crashing
    X_test[col] = X_test[col].astype(str).map(
        lambda val, le=le: le.transform([val])[0] if val in le.classes_ else -1
    )
    label_encoders[col] = le

# TODO: maybe try tuning these hyperparams later
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
}

results = {}
best_model_pipeline = None
best_cv_auc = 0.0
best_name = ""

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("---------------------------------")
for name, clf in models.items():
    print(f"Evaluating: {name}")

    # SMOTE is inside the pipeline so it never touches the validation fold during CV
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', clf)
    ])

    # only using X_train here — test set stays locked until the very end
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_auc = cv_scores.mean()

    results[name] = {"cv_auc_mean": round(cv_auc, 4)}
    print(f"  CV ROC-AUC: {cv_auc:.4f}")

    # picking the winner by CV score, not test score — that would be cheating
    if cv_auc > best_cv_auc:
        best_cv_auc = cv_auc
        best_model_pipeline = pipeline
        best_name = name

print("---------------------------------")
print(f"Best model: {best_name} (CV AUC = {best_cv_auc:.4f})\n")

# retrain the best model on the full training set before final evaluation
print("Training champion on full training set...")
best_model_pipeline.fit(X_train, y_train)

best_preds = best_model_pipeline.predict(X_test)
best_proba = best_model_pipeline.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, best_proba)

# this is the one and only time we look at the test set
results[best_name]["test_roc_auc"] = round(test_auc, 4)

print(f"Final Test ROC-AUC: {test_auc:.4f}")
print(f"Accuracy: {accuracy_score(y_test, best_preds):.4f}")
print(classification_report(y_test, best_preds, target_names=['Stayed', 'Churned']))

# tree models use feature_importances_, linear models use coef_ — we handle both
best_clf = best_model_pipeline.named_steps['classifier']
if hasattr(best_clf, 'feature_importances_'):
    importances = best_clf.feature_importances_
elif hasattr(best_clf, 'coef_'):
    # abs value because we care about magnitude, not direction
    importances = np.abs(best_clf.coef_[0])
else:
    importances = np.zeros(len(feature_names))

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

print("Top 10 churn drivers:")
print(feature_importance_df.head(10))

# TODO: add versioning here so we don't overwrite old models
from pathlib import Path
Path('artifacts').mkdir(exist_ok=True)

# saving the full pipeline so SMOTE and encoding are bundled with the model
with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(best_model_pipeline, f)

# UI team needs these to encode incoming inputs the same way we did
with open('artifacts/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

feature_importance_df.to_csv('artifacts/feature_importance.csv', index=False)

with open('artifacts/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

with open('artifacts/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\ndone, artifacts saved")