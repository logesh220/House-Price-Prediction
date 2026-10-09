
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# STEP 1: Load the dataset
df = pd.read_csv("data/cleaned_house_prices.csv")

target = "Price_INR_Lakhs"

df = df.dropna(subset=[target])
df = df[df[target] > 0]

# Create a folder for charts
os.makedirs("visualizations", exist_ok=True)

print("Dataset loaded successfully!")


# STEP 2: Property Price Distribution
plt.figure(figsize=(10, 6))

plt.hist(
    df[target],
    bins=50,
    edgecolor="black"
)

plt.title("Distribution of House Prices")
plt.xlabel("Property Price (Lakhs)")
plt.ylabel("Number of Properties")
plt.tight_layout()

plt.savefig("visualizations/price_distribution.png")
plt.show()

print("Price distribution chart saved.")


# STEP 3: Average Property Price by City
city_prices = (
    df.groupby("City")[target]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))

city_prices.plot(kind="bar", edgecolor="black")

plt.title("Average House Price by City")
plt.xlabel("City")
plt.ylabel("Average Price (Lakhs)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("visualizations/average_price_by_city.png")
plt.show()

print("City price chart saved.")


# STEP 4: Property Area vs. Price
plt.figure(figsize=(10, 6))

plt.scatter(
    df["Super_Area_SqFt"],
    df[target],
    alpha=0.4
)

plt.title("Property Area vs. House Price")
plt.xlabel("Super Area (Square Feet)")
plt.ylabel("Property Price (Lakhs)")
plt.tight_layout()

plt.savefig("visualizations/area_vs_price.png")
plt.show()

print("Area vs. price chart saved.")


# STEP 5: Random Forest Feature Importance
model = joblib.load("models/house_price_model.pkl")

preprocessor = model.named_steps["preprocessor"]
random_forest = model.named_steps["regressor"]

feature_names = preprocessor.get_feature_names_out()
importance_values = random_forest.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(15)

plt.figure(figsize=(11, 7))

plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1],
    edgecolor="black"
)

plt.title("Top 15 Features Influencing Model Predictions")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig("visualizations/feature_importance.png")
plt.close()

print("Feature importance chart saved.")


# STEP 6: Completion message
print("\nAll visualizations completed successfully!")
print("Charts are available in the visualizations folder.")

