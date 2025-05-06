import pandas as pd

df = pd.read_csv('data/raw/applications.csv')
print("=== Первые 5 строк ===")
print(df.head(), "\n")
print("=== Описательная статистика ===")
print(df.describe())
