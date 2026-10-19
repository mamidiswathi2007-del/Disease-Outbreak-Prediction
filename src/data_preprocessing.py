import pandas as pd
from pathlib import Path


def load_dataset(file_path=None):
    """Load the disease outbreak dataset."""

    if file_path is None:
        file_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "disease_data.csv"
        )

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    return df


def clean_dataset(df):
    """Remove duplicate rows and handle missing values."""

    df = df.copy()

    df = df.drop_duplicates()

    df = df.dropna(
        subset=["temperature", "rainfall", "cases", "outbreak"]
    )

    return df


if __name__ == "__main__":
    data = load_dataset()
    data = clean_dataset(data)

    print("Dataset loaded successfully!")
    print("Dataset shape:", data.shape)
    print("\nFirst five records:")
    print(data.head())
    print("\nMissing values:")
    print(data.isnull().sum())