# Smart Support Categorizer

[![Python Version](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

The Smart Support Categorizer is a machine learning-powered API designed to automatically classify support tickets into predefined categories: **Billing**, **Technical**, and **Other**. By automating the initial triage process, this tool helps streamline customer support workflows, ensuring that tickets are routed to the correct department efficiently.

This project uses a `TfidfVectorizer` for feature extraction and an ensemble of PyTorch neural networks for robust classification. The entire application is served as a RESTful API using FastAPI.

## Features

- **Automated Ticket Classification**: Automatically categorizes support tickets into 'Billing', 'Technical', or 'Other'.
- **Ensemble Model**: Utilizes an ensemble of neural networks for higher accuracy and more reliable predictions.
- **RESTful API**: Provides a simple and scalable API for easy integration with existing support desk software or other applications.
- **Ready-to-Use**: Includes scripts for data generation, model training, and API deployment.

## Directory Structure

```
└──smart-support-categorizer/
    ├── README.md
    ├── data_preparation.py
    ├── main.py
    ├── model_training.py
    ├── requirements.txt
    └── data/
        └── support_tickets.csv
```

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8+
- pip

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mukku27/smart-support-categorizer.git
    cd smart-support-categorizer
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare the dataset:**
    Run the `data_preparation.py` script to generate the `support_tickets.csv` file in the `data` directory.
    ```bash
    python data_preparation.py
    ```

5.  **Train the machine learning model:**
    Execute the `model_training.py` script to train the ensemble model. This will create a `model` directory and save the trained model as `ensemble_model.pth`.
    ```bash
    python model_training.py
    ```

### Running the API

Once the setup is complete, you can run the FastAPI application using Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Documentation

The API is self-documenting thanks to FastAPI. You can access the interactive API documentation (Swagger UI) by navigating to `http://127.0.0.1:8000/docs` in your browser.

### Endpoints

#### Health Check

-   **GET /**: Returns a welcome message to indicate that the API is running.

#### Prediction

-   **POST /predict**: Classifies a given support ticket text.

    **Request Body:**
    ```json
    {
      "text": "Mobile app  crashes frequently"
    }
    ```

    **Example Prediction JSON:**

    Here is an example of the JSON output for a prediction request:
    ```json
    {
      "prediction": "Technical",
      "confidence": 0.9686,
      "probabilities": {
        "Technical": 0.9686,
        "Billing": 0.0181,
        "Other": 0.0133
      }
    }
    ```

## How It Works

1.  **Data Preparation**: The `data_preparation.py` script generates a synthetic dataset of support tickets with 'Billing', 'Technical', and 'Other' labels.

2.  **Model Training**:
    -   The text is preprocessed by converting it to lowercase and removing non-alphabetic characters.
    -   A `TfidfVectorizer` is used to convert the text data into numerical features.
    -   An ensemble of four `TextClassifier` neural networks is trained on the vectorized data.
    -   The trained model, vectorizer, and label map are saved to a single file.

3.  **API Server**:
    -   The `main.py` file sets up a FastAPI application.
    -   On startup, the application loads the trained model and associated components.
    -   The `/predict` endpoint accepts a POST request with the ticket text, preprocesses it, and returns the predicted category along with a confidence score and the probabilities for each category.
