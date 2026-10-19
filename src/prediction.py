import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "disease_model.pkl"
)


def load_model():
    """Load the trained machine learning model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_outbreak(temperature, rainfall, cases):
    """Predict the outbreak class for the given input."""

    model = load_model()

    input_data = pd.DataFrame(
        [[temperature, rainfall, cases]],
        columns=["temperature", "rainfall", "cases"]
    )

    prediction = model.predict(input_data)

    return int(prediction[0])