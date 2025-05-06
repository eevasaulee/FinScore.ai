import os, joblib, pandas as pd, matplotlib.pyplot as plt, shap

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[DEBUG] Model dumped to {os.path.abspath(path)}")


def load_model(path: str):
    return joblib.load(path)

def save_dataframe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def save_shap_summary_plot(explainer, X: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(output_path)
    plt.close()
