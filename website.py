from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "disease_data.csv"
MODEL_PATH = BASE_DIR / "models" / "disease_model.pkl"


def load_dataset():
    if not DATA_PATH.exists():
        return pd.DataFrame(
            columns=["temperature", "rainfall", "cases", "outbreak"]
        )

    return pd.read_csv(DATA_PATH)


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    df = load_dataset()

    total_records = len(df)
    class_0 = 0
    class_1 = 0

    if "outbreak" in df.columns:
        class_0 = int((df["outbreak"] == 0).sum())
        class_1 = int((df["outbreak"] == 1).sum())

    return render_template(
        "dashboard.html",
        total_records=total_records,
        class_0=class_0,
        class_1=class_1,
        temperature_avg=round(df["temperature"].mean(), 2)
        if total_records else 0,
        rainfall_avg=round(df["rainfall"].mean(), 2)
        if total_records else 0,
        cases_avg=round(df["cases"].mean(), 2)
        if total_records else 0,
    )


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result = None
    error = None

    if request.method == "POST":
        try:
            temperature = float(request.form["temperature"])
            rainfall = float(request.form["rainfall"])
            cases = float(request.form["cases"])

            if not -50 <= temperature <= 70:
                raise ValueError(
                    "Temperature must be between -50 and 70 °C."
                )

            if rainfall < 0 or cases < 0:
                raise ValueError(
                    "Rainfall and reported cases cannot be negative."
                )

            if not cases.is_integer():
                raise ValueError(
                    "Reported cases must be a whole number."
                )

            model = load_model()

            if model is None:
                raise FileNotFoundError(
                    "Model not found. Run train_model.py first."
                )

            input_data = pd.DataFrame(
                [[temperature, rainfall, cases]],
                columns=["temperature", "rainfall", "cases"]
            )

            predicted_class = int(model.predict(input_data)[0])

            result = {
                "class": predicted_class,
                "temperature": temperature,
                "rainfall": rainfall,
                "cases": int(cases),
            }

        except (ValueError, KeyError, FileNotFoundError) as exc:
            error = str(exc)

        except Exception:
            error = (
                "Prediction could not be completed. "
                "Please check the model and input values."
            )

    return render_template(
        "prediction.html",
        result=result,
        error=error,
    )


@app.route("/analytics")
def analytics():
    df = load_dataset()

    if df.empty:
        labels = []
        class_counts = []
        temperature_values = []
        rainfall_values = []
        cases_values = []
    else:
        counts = df["outbreak"].value_counts().sort_index()
        labels = [str(x) for x in counts.index.tolist()]
        class_counts = counts.tolist()

        temperature_values = df["temperature"].tolist()
        rainfall_values = df["rainfall"].tolist()
        cases_values = df["cases"].tolist()

    return render_template(
        "analytics.html",
        labels=labels,
        class_counts=class_counts,
        temperature_values=temperature_values,
        rainfall_values=rainfall_values,
        cases_values=cases_values,
    )


@app.route("/dataset")
def dataset():
    df = load_dataset()
    records = df.fillna("").to_dict(orient="records")

    return render_template(
        "dataset.html",
        records=records,
        columns=df.columns.tolist(),
        total_records=len(df),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)