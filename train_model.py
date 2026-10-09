
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# STEP 1: Load the cleaned dataset
df = pd.read_csv("data/cleaned_house_prices.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# STEP 2: Define the target variable
target = "Price_INR_Lakhs"

# Remove rows without a valid target price
df = df.dropna(subset=[target])
df = df[df[target] > 0]


# STEP 3: Separate input features and target
# Property_ID is an identifier, so we don't use it for prediction.
columns_to_remove = [target]

if "Property_ID" in df.columns:
    columns_to_remove.append("Property_ID")

X = df.drop(columns=columns_to_remove)
y = df[target]

print("\nInput features:")
print(X.columns.tolist())

print("\nTarget variable:", target)


# STEP 4: Identify numerical and categorical columns
numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)


# STEP 5: Prepare numerical features
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# STEP 6: Prepare categorical features
categorical_pipeline = Pipeline(
    steps=[
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
    ]
)


# STEP 7: Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)


# STEP 8: Create the updated Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# STEP 9: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# STEP 10: Train the model
print("\nTraining the Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")


# STEP 11: Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-----------------------------")
print(f"Mean Absolute Error: {mae:.2f} lakhs")
print(f"R² Score: {r2:.4f}")


# STEP 12: Save the trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/house_price_model.pkl")

print("\nTrained model saved successfully!")
print("Location: models/house_price_model.pkl")

