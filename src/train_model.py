import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import shap
from src.utils import save_model, save_dataframe, save_shap_summary_plot

def main():
    df = pd.read_csv('data/raw/applications.csv').dropna()
    save_dataframe(df, 'data/processed/app_cleaned.parquet')

    X = df.drop(columns=['default'])
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,
    random_state=42,
    stratify=y
)

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:,1]
    print('ROC AUC:', roc_auc_score(y_test, y_pred))

    save_model(model, 'models/gbc_model.pkl')
    explainer = shap.Explainer(model, X_train)
    save_shap_summary_plot(explainer, X_test, 'models/shap_summary.png')

if __name__ == '__main__':
    main()
