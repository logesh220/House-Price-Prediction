import pandas as pd

df = pd.read_csv("data/cleaned_house_prices.csv")

columns = [
"City",
"Locality_Type",
"Property_Type",
"Furnishing_Status"
]

for column in columns:
    print()
    print(column + ":")
    categories = df[column].dropna().astype(str).unique()
    print(sorted(categories))
