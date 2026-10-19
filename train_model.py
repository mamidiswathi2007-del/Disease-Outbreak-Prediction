import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Project paths
DATA_PATH = Path("data/disease_data.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "disease_model.pkl"

# Check dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found: data/disease_data.csv"
    )

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())

# Expected columns in the dataset
features = ["temperature", "rainfall", "cases"]
target = "outbreak"

missing = [col for col in features + [target] if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        "Please check disease_data.csv."
    )

# Prepare data
X = df[features]
y = df[target]

# Check target values
if y.nunique() < 2:
    raise ValueError(
        "The outbreak column must contain at least two classes."
    )

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("\nModel training completed!")
print("Test accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")