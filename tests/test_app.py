import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Import the functions to be tested from your Streamlit app script
# FIX: The import path is changed to point inside the 'src' package
from src.app import collect_logs, get_processed_data, establish_baseline


# Use a module-scoped fixture to run the expensive data generation only once
@pytest.fixture(scope="module")
def raw_logs_df():
    """Fixture to provide a raw DataFrame of collected logs."""
    return collect_logs(num_logs=500)


@pytest.fixture(scope="module")
def processed_data():
    """Fixture to provide the fully processed data and ML artifacts."""
    # Since get_processed_data is cached, this is efficient.
    # We call it once and reuse the results across multiple tests.
    df, features, scaler, shap_values = get_processed_data()
    return {
        "df": df,
        "features": features,
        "scaler": scaler,
        "shap_values": shap_values
    }


# --- Tests for collect_logs ---

def test_collect_logs_shape_and_columns(raw_logs_df):
    """Test if collect_logs returns a DataFrame with the correct shape and columns."""
    assert isinstance(raw_logs_df, pd.DataFrame)
    assert raw_logs_df.shape[0] == 500
    expected_cols = ['timestamp', 'user_id', 'ip', 'bandwidth_usage', 'packet_count', 'event_type']
    assert all(col in raw_logs_df.columns for col in expected_cols)


def test_collect_logs_anomaly_simulation(raw_logs_df):
    """Test if the anomaly simulation correctly increases bandwidth for suspicious events."""
    avg_suspicious_bw = raw_logs_df[raw_logs_df['event_type'] == 'suspicious']['bandwidth_usage'].mean()
    avg_normal_bw = raw_logs_df[raw_logs_df['event_type'] == 'normal']['bandwidth_usage'].mean()
    # The multiplier is 3, so the average should be significantly higher
    assert avg_suspicious_bw > avg_normal_bw * 2.5


# --- Tests for get_processed_data ---

def test_get_processed_data_return_types(processed_data):
    """Test if get_processed_data returns objects of the correct types."""
    assert isinstance(processed_data["df"], pd.DataFrame)
    assert isinstance(processed_data["features"], list)
    assert isinstance(processed_data["scaler"], StandardScaler)
    assert isinstance(processed_data["shap_values"], np.ndarray)


def test_get_processed_data_feature_engineering(processed_data):
    """Test if feature engineering columns are correctly added."""
    df = processed_data["df"]
    assert 'ip_hashed' in df.columns
    assert 'packet_rate' in df.columns
    assert 'login_failure' in df.columns
    assert 'anomaly' in df.columns
    assert 'anomaly_score' in df.columns

    # Check if login_failure is binary
    assert df['login_failure'].isin([0, 1]).all()

    # Check if anomaly labels are correct
    assert df['anomaly'].isin(['Normal', 'Anomaly']).all()


def test_ip_hashing_consistency(processed_data):
    """Test that IP hashing is consistent for the same IP."""
    df = processed_data["df"]
    # Find a user with multiple log entries
    user_with_multiple_logs = df['user_id'].value_counts().idxmax()
    user_df = df[df['user_id'] == user_with_multiple_logs]

    # All hashed IPs for this user (who should have a consistent IP) should be the same
    if not user_df.empty:
        assert user_df['ip_hashed'].nunique() == 1


# --- Tests for establish_baseline ---

def test_establish_baseline_calculation():
    """Test the baseline calculation with a predictable, manually created DataFrame."""
    data = {
        'bandwidth_usage': [100, 110, 90, 105, 95, 500],
        'anomaly': ['Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Anomaly']
    }
    df = pd.DataFrame(data)

    # Calculation should only use the 'Normal' values: [100, 110, 90, 105, 95]
    normal_data = np.array([100, 110, 90, 105, 95])
    expected_mean = normal_data.mean()  # 100.0
    expected_std = normal_data.std(ddof=1)  # ddof=1 for sample std dev, matching pandas .std()

    baseline = establish_baseline(df)

    assert baseline['mean'] == pytest.approx(expected_mean)
    assert baseline['std'] == pytest.approx(expected_std)
    assert baseline['threshold'] == pytest.approx(expected_mean + 3 * expected_std)


def test_establish_baseline_with_corrected_labels():
    """Test that 'Normal (Corrected)' labels are included in the baseline."""
    data = {
        'bandwidth_usage': [100, 110, 90, 200, 500],
        'anomaly': ['Normal', 'Normal', 'Normal', 'Normal (Corrected)', 'Anomaly']
    }
    df = pd.DataFrame(data)

    # Calculation should use [100, 110, 90, 200]
    normal_data = np.array([100, 110, 90, 200])
    expected_mean = normal_data.mean()  # 125.0

    baseline = establish_baseline(df)

    assert baseline['mean'] == pytest.approx(expected_mean)
