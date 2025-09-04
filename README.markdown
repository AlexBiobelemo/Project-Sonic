# VPN Log Anomaly Detection Simulation

This project is a high-fidelity simulation of a real-time anomaly detection system for VPN logs. It uses an unsupervised machine learning model to identify suspicious activity and provides tools for model explainability and interactive feedback, all packaged within a user-friendly Streamlit web application.

## 🚀 Key Features

- **Realistic Data Simulation**: Generates a dataset of VPN logs with a consistent user-to-IP mapping, ensuring data quality for model training.
- **Unsupervised Anomaly Detection**: Employs an Isolation Forest algorithm to identify outliers in bandwidth usage, packet rates, and login failures without needing pre-labeled data.
- **Model Explainability (XAI)**: Integrates SHAP (SHapley Additive exPlanations) to explain why a specific log was flagged as an anomaly, providing crucial insights for security analysts.
- **Interactive Dashboard**: A dynamic Streamlit interface allows users to filter data by time, inspect anomalies, and visualize data against a dynamically calculated baseline.
- **Human-in-the-Loop Feedback**: Features an interactive "Model Refinement" section where users can mark false positives, re-label data, and see the dashboard update in real-time to simulate a model retraining loop.
- **Comprehensive Test Suite**: Includes a robust suite of pytest tests to ensure the reliability and correctness of the core data processing and machine learning logic.

## 🛠️ Technology Stack

- **Backend & ML**: Python, Pandas, Scikit-learn
- **Web Framework**: Streamlit
- **Data Visualization**: Plotly
- **Model Explainability**: SHAP
- **Testing**: Pytest

## 📂 Final Project Structure

The project is structured as a standard Python package to ensure modularity and reliable import resolution between the application and test code.

```
vpn_anomaly_project/
├── src/
│   ├── __init__.py      # Marks 'src' as a code package
│   └── app.py           # The main Streamlit application logic
│
├── tests/
│   ├── __init__.py      # Marks 'tests' as a test package
│   └── test_app.py      # The pytest test suite
│
├── pyproject.toml       # Modern Python project configuration
├── pytest.ini           # Directs pytest to find the 'src' module
├── requirements.txt     # All project dependencies
└── README.md            # This file
```

## ⚙️ Setup and Execution

Follow these steps to set up your environment and run the application.

### 1. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies. From the project's root directory, run:

```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

With the virtual environment active, install all required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Tests (Recommended)

Before launching the app, verify that everything is configured correctly by running the test suite from the project root directory:

```bash
pytest -v
```

You should see all 7 tests pass without any errors or warnings.

### 4. Launch the Application

Run the following command from the project root directory to start the Streamlit server:

```bash
streamlit run src/app.py
```

Your web browser should automatically open to the application's URL.