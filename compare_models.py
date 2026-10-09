
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import mean_absolute_error, r2_score


# STEP 1: Load the dataset
print("Loading dataset...")

df = pd.read_csv("data/cleaned_house_prices.csv")

target = "Price_INR_Lakhs"

# Remove rows with invalid target values
df = df.dropna(subset=[target])
df = df[df[target] > 0]

# STEP 2: Separate input features and target
columns_to_remove = [target]

if "Property_ID" in df.columns:
    columns_to_remove.append("Property_ID")

X = df.drop(columns=columns_to_remove)
y = df[target]

# STEP 3: Identify numerical and categorical columns
numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print(f"Total records: {len(df)}")
print(f"Numerical features: {len(numerical_columns)}")
print(f"Categorical features: {len(categorical_columns)}")

# STEP 4: Prepare preprocessing pipelines
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer(transformers=[
    (
        "numerical",
        numerical_pipeline,
        numerical_columns
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])

# STEP 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# STEP 6: Define four machine learning algorithms
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=20,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
}

# STEP 7: Train and evaluate each model
results = []

for model_name, regressor in models.items():

    print(f"\nTraining {model_name}...")

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE (Lakhs)": round(mae, 2),
        "R2 Score": round(r2, 4)
    })

    print(f"MAE: {mae:.2f} lakhs")
    print(f"R2 Score: {r2:.4f}")

# STEP 8: Display comparison results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE (Lakhs)",
    ascending=True
)

print("\n" + "=" * 55)
print("MACHINE LEARNING MODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

best_model = results_df.iloc[0]

print("\nBest model based on lowest MAE:")
print(best_model["Model"])
print(f"MAE: {best_model['MAE (Lakhs)']} lakhs")
print(f"R2 Score: {best_model['R2 Score']}")

print("\nComparison completed successfully!")

