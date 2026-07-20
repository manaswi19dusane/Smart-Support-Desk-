# Smart Support Desk

Smart Support Desk is a customer support ticket classifier. It uses a trained
machine learning model to predict the category of a support ticket from the
ticket text.

## Project Structure

```text
smart-support-desk/
  app/
    main.py              # FastAPI backend
    schemas.py           # Request and response models
    ml/
      model.joblib       # Trained classifier
      vectorizer.joblib  # Trained TF-IDF vectorizer
  ml_training/
    train_model.py       # Script used to train the ML model
  requirements.txt       # Python dependencies
```

## Step 1: Open the Project Folder

Run this command from your terminal:

```bash
cd smart-support-desk
```

## Step 2: Create and Activate a Virtual Environment

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

If you are using Command Prompt instead:

```bash
venv\Scripts\activate
```

## Step 3: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

## Step 4: Train the ML Model

Run the training script:

```bash
python ml_training/train_model.py
```

This creates two files inside `app/ml/`:

```text
app/ml/model.joblib
app/ml/vectorizer.joblib
```

## Step 5: Start the FastAPI Backend

Run this command from the `smart-support-desk` folder:

```bash
uvicorn app.main:app --reload
```

The API will start at:

```text
http://127.0.0.1:8000
```

## Step 6: Check the Health Endpoint

Open this URL in your browser:

```text
http://127.0.0.1:8000/health
```

You should see:

```json
{"status": "ok"}
```

## Step 7: Test the Ticket Classifier

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

In the docs:

1. Open `POST /tickets`.
2. Click `Try it out`.
3. Enter a request body like this:

```json
{
  "text": "I cannot log in to my account after resetting my password."
}
```

4. Click `Execute`.
5. The response will include the predicted ticket category.

Example response:

```json
{
  "text": "I cannot log in to my account after resetting my password.",
  "predicted_category": "account_access"
}
```
