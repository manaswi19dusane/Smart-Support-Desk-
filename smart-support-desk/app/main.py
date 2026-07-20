from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI

from app.schemas import TicketIn, TicketOut


# Find the folder that contains this file: app/
APP_DIR = Path(__file__).resolve().parent

# The trained ML files are stored in app/ml/.
MODEL_PATH = APP_DIR / "ml" / "model.joblib"
VECTORIZER_PATH = APP_DIR / "ml" / "vectorizer.joblib"

# These variables will hold the loaded model and vectorizer.
# They start as None and are filled once when the FastAPI app starts.
model = None
vectorizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML files once when the server starts."""
    global model, vectorizer

    # joblib.load reads the saved Python objects from disk.
    # Loading them here avoids doing slow file work for every API request.
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # yield lets FastAPI start serving requests after setup is complete.
    yield


# Create the FastAPI application.
# The lifespan function above runs once at startup.
app = FastAPI(
    title="Smart Support Desk API",
    description="A simple API that classifies customer support tickets.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """Return a small response to confirm the API server is running."""
    return {"status": "ok"}


@app.post("/tickets", response_model=TicketOut)
def classify_ticket(ticket: TicketIn):
    """Predict the category for one support ticket."""
    # Apply the same minimal cleaning used during training.
    cleaned_text = ticket.text.lower()

    # Convert the text into TF-IDF numbers using the saved vectorizer.
    text_features = vectorizer.transform([cleaned_text])

    # Ask the saved model to predict the category.
    predicted_category = model.predict(text_features)[0]

    # Return the original text plus the predicted category.
    return TicketOut(text=ticket.text, predicted_category=predicted_category)
