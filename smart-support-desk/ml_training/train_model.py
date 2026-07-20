from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# This script lives in ml_training/, so the project root is one folder above it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Update this path if you want to train on a different CSV file.
DATASET_PATH = PROJECT_ROOT / "support-ticket-classification.csv"

# The trained files will be saved here for the app to load later.
MODEL_DIR = PROJECT_ROOT / "app" / "ml"
MODEL_PATH = MODEL_DIR / "model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the CSV file and make sure it has the columns we need."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {csv_path}")

    data = pd.read_csv(csv_path)

    # The Phase 1 requirement expects a "category" column.
    # The current sample dataset uses "label", so we rename it for compatibility.
    if "category" not in data.columns and "label" in data.columns:
        data = data.rename(columns={"label": "category"})

    required_columns = {"text", "category"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "Dataset must contain columns named 'text' and 'category'. "
            f"Missing: {sorted(missing_columns)}"
        )

    # Keep only the columns used for training and remove empty rows.
    data = data[["text", "category"]].dropna()
    return data


def main() -> None:
    """Train and save a ticket category classifier."""
    # 1. Load the dataset from CSV.
    data = load_dataset(DATASET_PATH)

    # 2. Clean the ticket text minimally by converting it to lowercase.
    data["text"] = data["text"].astype(str).str.lower()

    # 3. Split the data into features (X) and labels (y).
    X = data["text"]
    y = data["category"]

    # 4. Split into training and testing sets.
    #    80% is used for training and 20% is held back for evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 5. Convert text into numeric TF-IDF features that the model can understand.
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 6. Train a Logistic Regression classifier on the vectorized text.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # 7. Predict categories for the test set.
    y_pred = model.predict(X_test_tfidf)

    # 8. Print common classification metrics.
    #    Weighted averages work well when categories have different sample counts.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("Model evaluation results")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # 9. Create the app/ml folder if it does not already exist.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 10. Save the trained model and vectorizer so the app can reuse them later.
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
