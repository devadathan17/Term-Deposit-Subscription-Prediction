import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
import warnings
warnings.filterwarnings("ignore")

path = r"D:\LGBM Project\bank_data.csv"
df = pd.read_csv(path)
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

cat_cols = df.select_dtypes(include='object').columns.tolist()
df_encoded = df.copy()

le = LabelEncoder()
df_encoded['education'] = le.fit_transform(df_encoded['education'])

month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
df_encoded['month'] = df_encoded['month'].map(month_map)

df_encoded['y'] = df_encoded['y'].map({'yes': 1, 'no': 0})

binary_cols = ['default', 'housing', 'loan']
binary_encoder = ce.BinaryEncoder(cols=binary_cols)
binary_encoded = binary_encoder.fit_transform(df_encoded[binary_cols])
df_encoded.drop(columns=binary_cols, inplace=True)
df_encoded = pd.concat([df_encoded, binary_encoded], axis=1)

onehot_cols = ['job', 'contact', 'marital', 'poutcome']
df_encoded = pd.get_dummies(df_encoded, columns=onehot_cols, drop_first=True)
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Set Class Distribution:\n", y_train.value_counts())

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
initial_spw = neg_count / pos_count
print(f"Initial scale_pos_weight: {initial_spw:.2f}")

print("\nApplying RFE for Feature Selection...")
base_model = lgb.LGBMClassifier(random_state=42)
rfe_selector = RFE(estimator=base_model, n_features_to_select=25, step=1)
rfe_selector = rfe_selector.fit(X_train, y_train)

selected_features = X_train.columns[rfe_selector.support_]
print("Selected Features:", list(selected_features))

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

param_grid = {
    'num_leaves': [15, 31],
    'max_depth': [-1, 5, 10],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 200],
    'scale_pos_weight': [initial_spw, initial_spw * 0.5, initial_spw * 1.5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=lgb.LGBMClassifier(random_state=42),
    param_grid=param_grid,
    scoring='recall',
    cv=cv,
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X_train_selected, y_train)
print("Best Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_selected)
y_proba = best_model.predict_proba(X_test_selected)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

custom_threshold = 0.59
y_pred = (y_proba >= custom_threshold).astype(int)

print("\nClassification Report with Custom Threshold:\n")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))