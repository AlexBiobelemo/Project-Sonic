import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import hashlib
import datetime
import json
import shap

# Streamlit app configuration
st.set_page_config(page_title="VPN Log Anomaly Detection Simulation", layout="wide")


# --- Core Logic Functions ---

@st.cache_data
def collect_logs(num_logs=1000, num_users=50):
    """
    Simulate VPN logs with a realistic user-to-IP mapping.
    This ensures a user is consistently associated with a single IP address.
    """
    np.random.seed(42)

    # 1. Create a stable mapping of users to IPs to make the simulation realistic.
    #    Each user is assigned a consistent IP address.
    user_ips = {user_id: f"192.168.1.{np.random.randint(2, 255)}" for user_id in range(1, num_users + 1)}

    # 2. Generate log events by first choosing a user for each event.
    log_user_ids = np.random.randint(1, num_users + 1, num_logs)

    # 3. Look up the IP for each chosen user from the stable mapping.
    log_ips = [user_ips[user_id] for user_id in log_user_ids]

    data = {
        'timestamp': pd.date_range(start='2025-09-04 06:00:00', periods=num_logs, freq='s'),
        # FIX: Changed 'S' to 's' to resolve FutureWarning
        'user_id': log_user_ids,
        'ip': log_ips,
        'bandwidth_usage': np.random.normal(100, 20, num_logs),  # Normal: ~100 MB/s
        'packet_count': np.random.normal(1000, 200, num_logs),  # Normal: ~1000 packets/s
        'event_type': np.random.choice(['normal', 'auth_failure', 'suspicious'], num_logs, p=[0.90, 0.05, 0.05])
    }
    df = pd.DataFrame(data)

    # Simulate anomalies
    df.loc[df['event_type'] == 'suspicious', 'bandwidth_usage'] *= 3
    df.loc[df['event_type'] == 'suspicious', 'packet_count'] *= 3
    df.loc[df['event_type'] == 'auth_failure', 'bandwidth_usage'] *= 0.5
    return df


@st.cache_data
def get_processed_data():
    """
    Runs the full data processing and ML pipeline.
    This is cached to prevent re-computation on every user interaction.
    """
    logs = collect_logs()
    df = logs.drop_duplicates().dropna()
    df['ip_hashed'] = df['ip'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
    df['packet_rate'] = df['packet_count'] / 10
    df['login_failure'] = (df['event_type'] == 'auth_failure').astype(int)

    features = ['bandwidth_usage', 'packet_rate', 'login_failure']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)

    predictions = model.predict(X_scaled)
    df['anomaly_score'] = model.decision_function(X_scaled)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    df['anomaly'] = pd.Series(predictions, index=df.index).map({1: 'Normal', -1: 'Anomaly'})
    df['shap_importance'] = [sum(abs(val)) for val in shap_values]

    return df.copy(), features, scaler, shap_values


def establish_baseline(df, feature='bandwidth_usage'):
    """Calculate historical baseline for normal behavior, excluding anomalies."""
    normal_df = df[df['anomaly'].str.contains('Normal')]
    if normal_df.empty:
        return {'mean': 0, 'std': 0, 'threshold': 0}
    baseline = {
        'mean': normal_df[feature].mean(),
        'std': normal_df[feature].std(),
        'threshold': normal_df[feature].mean() + 3 * normal_df[feature].std()
    }
    return baseline


# --- UI Functions ---

def visualize_and_respond(df, baseline, features, shap_values):
    """Visualize anomalies and trigger responses."""
    st.subheader("Log Analysis Dashboard")

    min_date = df['timestamp'].min().to_pydatetime()
    max_date = df['timestamp'].max().to_pydatetime()

    date_range = st.slider(
        "Select Time Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    filtered_df = df[(df['timestamp'] >= date_range[0]) & (df['timestamp'] <= date_range[1])]

    st.write("**Filtered Logs**")
    st.dataframe(filtered_df[['timestamp', 'user_id', 'ip_hashed', 'bandwidth_usage', 'packet_rate', 'anomaly',
                              'anomaly_score']].head(10))

    fig = px.scatter(filtered_df, x='timestamp', y='bandwidth_usage', color='anomaly',
                     title="Bandwidth Usage with Anomalies",
                     color_discrete_map={'Normal': 'blue', 'Anomaly': 'red', 'Normal (Corrected)': 'green'},
                     hover_data=['user_id', 'ip_hashed', 'packet_rate'])
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"**Baseline Bandwidth**: Mean = {baseline['mean']:.2f} MB/s, Threshold = {baseline['threshold']:.2f} MB/s")
    fig_baseline = px.line(filtered_df, x='timestamp', y='bandwidth_usage', title="Bandwidth vs. Baseline")
    fig_baseline.add_hline(y=baseline['threshold'], line_dash="dash", line_color="red",
                           annotation_text="Anomaly Threshold")
    st.plotly_chart(fig_baseline, use_container_width=True)

    anomalies_in_view = filtered_df[filtered_df['anomaly'] == 'Anomaly']
    if not anomalies_in_view.empty:
        st.subheader("Top Anomaly Explanation")
        top_anomaly = anomalies_in_view.sort_values('anomaly_score').iloc[0]

        top_anomaly_original_index = top_anomaly.name
        top_anomaly_iloc = df.index.get_loc(top_anomaly_original_index)

        shap_data = pd.DataFrame({'Feature': features, 'SHAP Value': shap_values[top_anomaly_iloc]})
        fig_shap = px.bar(shap_data, x='SHAP Value', y='Feature', orientation='h',
                          title=f"SHAP Explanation for Anomaly at {top_anomaly['timestamp']}")
        st.plotly_chart(fig_shap, use_container_width=True)

    if st.button("Block Anomalous IPs in selected time range"):
        anomalous_ips = filtered_df[filtered_df['anomaly'] == 'Anomaly']['ip_hashed'].unique()
        if anomalous_ips.size > 0:
            st.warning(f"Simulating IP block for: {', '.join(anomalous_ips[:5])}...")
            with open('blocked_ips.json', 'a') as f:
                json.dump({'timestamp': str(datetime.datetime.now()), 'ips': list(anomalous_ips)}, f)
                f.write('\n')
        else:
            st.info("No anomalies to block in the selected time range.")


def document_and_refine(df_to_refine):
    """Save results and handle the interactive model refinement simulation."""
    st.header("Audit & Refinement")

    if st.button("Save Current Analysis for Audit"):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"vpn_log_analysis_{timestamp}.csv"
        df_to_refine.to_csv(filename)
        st.success(f"Results saved as {filename}")

    st.subheader("Model Refinement Simulation")
    st.markdown("""
    Provide feedback to the model. Indicate how many of the detected anomalies you believe are false positives. 
    The simulation will re-label the **least severe** anomalies as 'Normal (Corrected)' and update the dashboard.
    """)

    anomalies_to_refine = df_to_refine[df_to_refine['anomaly'] == 'Anomaly']
    max_correction = len(anomalies_to_refine)

    false_positives = st.number_input(
        "Number of false positives to correct",
        min_value=0,
        max_value=max_correction,
        value=0,
        step=1,
        help=f"You can correct up to {max_correction} anomalies currently shown."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Apply Refinement", disabled=(false_positives == 0)):
            to_correct_indices = anomalies_to_refine.nsmallest(false_positives, 'anomaly_score').index

            refined_df = df_to_refine.copy()
            refined_df.loc[to_correct_indices, 'anomaly'] = 'Normal (Corrected)'
            st.session_state.refined_df = refined_df
            st.success(f"Applied feedback: {false_positives} anomalies re-labeled.")
            st.rerun()

    with col2:
        if st.button("Reset to Original Analysis", disabled=('refined_df' not in st.session_state)):
            del st.session_state.refined_df
            st.info("Analysis has been reset to the original model output.")
            st.rerun()


def main():
    """Main function to run the Streamlit app."""
    st.title("VPN Log Anomaly Detection Simulation")
    st.write("Running analysis... this may take a moment on first load.")

    original_df, features, _, shap_values = get_processed_data()

    if 'refined_df' in st.session_state:
        active_df = st.session_state.refined_df
        st.info("Displaying analysis with your feedback applied.")
    else:
        active_df = original_df

    baseline = establish_baseline(active_df)
    visualize_and_respond(active_df, baseline, features, shap_values)
    document_and_refine(active_df)


if __name__ == "__main__":
    main()

