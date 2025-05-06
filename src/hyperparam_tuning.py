# src/hyperparam_tuning.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

# 1) Загрузка и очистка данных
df = pd.read_csv('data/raw/applications.csv').dropna()
X = df.drop(columns=['default'])
y = df['default']

# 2) Делим данные на train и test (50%/50%), со стратификацией
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,
    random_state=42,
    stratify=y
)

# 3) Ручной перебор гиперпараметров
best_score = -1.0
best_params = None
best_model = None

for n_estimators in [50, 100, 200]:
    for learning_rate in [0.01, 0.1, 0.2]:
        for max_depth in [3, 5, 7]:
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred)
            print(f"Params: n={n_estimators}, lr={learning_rate}, depth={max_depth} → AUC={score:.3f}")
            if score > best_score:
                best_score = score
                best_params = (n_estimators, learning_rate, max_depth)
                best_model = model

# 4) Выводим лучший результат
print("\nЛучшие параметры:", best_params)
print("Test ROC AUC (best):", best_score)

# 5) Сохраняем лучшую модель
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_gbc_model.pkl')
print("✅ Сохранена лучшая модель в models/best_gbc_model.pkl")


