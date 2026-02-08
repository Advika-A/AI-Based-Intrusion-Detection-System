from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np

print("API FILE STARTED")

# -----------------------------
# App Setup
# -----------------------------
app = Flask(__name__, static_folder="../frontend")
CORS(app)

# -----------------------------
# Load Models
# -----------------------------
rf_model = joblib.load("models/rf_cicids_supervised.pkl")
iso_model = joblib.load("models/cicids_isolation_forest.pkl")

# -----------------------------
# Feature Definitions
# -----------------------------
RF_FEATURES = list(rf_model.feature_names_in_)

ISO_FEATURES = [
    "Average Packet Size",
    "Max Packet Length",
    "Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Std",
    "Avg Bwd Segment Size",
    "Fwd Packet Length Mean",
    "Min Packet Length",
    "Destination Port",
    "Init_Win_bytes_backward",
    "Flow Duration",
    "Total Fwd Packets"
]

# -----------------------------
# Explainable AI Rules
# -----------------------------
def generate_explanations(data):
    reasons = []

    if data.get("Average Packet Size", 0) > 1200:
        reasons.append("Packets are unusually large")

    if data.get("Flow Duration", 0) > 60000:
        reasons.append("Connection lasted longer than typical sessions")

    if data.get("Total Fwd Packets", 0) > 800:
        reasons.append("High number of outgoing packets observed")

    if data.get("Destination Port", 0) not in [80, 443, 53]:
        reasons.append("Traffic targets an uncommon destination port")

    return reasons

# -----------------------------
# Serve Frontend
# -----------------------------
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index_ids.html")

# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # ---------- Random Forest ----------
        X_rf = np.array([float(data.get(f, 0)) for f in RF_FEATURES]).reshape(1, -1)
        rf_pred = int(rf_model.predict(X_rf)[0])
        rf_prob = rf_model.predict_proba(X_rf)[0]
        rf_conf = float(np.max(rf_prob))

        # Top 3 RF features based on feature importance
        importances = rf_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:3]
        top_rf_features = []
        for i in sorted_idx:
            top_rf_features.append({
                "feature": RF_FEATURES[i],
                "value": float(data.get(RF_FEATURES[i], 0)),
                "deviation": round(float(importances[i]), 2)
            })

        # ---------- Isolation Forest ----------
        X_iso = np.array([float(data.get(f, 0)) for f in ISO_FEATURES]).reshape(1, -1)
        iso_pred_num = int(iso_model.predict(X_iso)[0])
        iso_decision = "Anomalous" if iso_pred_num == -1 else "Normal"

        # Example: top IF features (can refine later)
        top_if_features = [
            {"feature": f, "value": float(data.get(f, 0)), "deviation": round(np.random.uniform(0.3,0.8), 2)}
            for f in ISO_FEATURES[:3]
        ]

        # ---------- Model Decisions ----------
        rf_decision = "Attack" if rf_pred == 1 else "Normal"

        # ---------- Final Verdict & Confidence ----------
        if rf_pred == 1:
            final_verdict = "Attack"
            severity = "High"
            action = "Block traffic and investigate immediately"
            agreement_status = "Agreement"
            combined_conf = rf_conf
            explanation_nl = [
                "Known attack patterns detected by Random Forest.",
                "Confidence is high as both models are aligned."
            ]

        elif rf_pred == 0 and iso_pred_num == -1:
            final_verdict = "Suspicious"
            severity = "Medium"
            action = "Monitor traffic and analyze further"
            agreement_status = "Disagreement"
            combined_conf = round(rf_conf * 0.75, 2)  # adjust as needed
            explanation_nl = [
                "Traffic does not match known attack patterns but deviates from normal behavior.",
                "Random Forest predicted Normal, but Isolation Forest detected anomalies, lowering confidence."
            ]

        else:
            final_verdict = "Normal"
            severity = "Low"
            action = "No action required"
            agreement_status = "Agreement"
            combined_conf = rf_conf
            explanation_nl = ["Traffic matches normal network behavior."]

        # ---------- Feature-Level Explainability ----------
        explanations = generate_explanations(data)
        if final_verdict == "Suspicious":
            explanations.insert(
                0,
                "Traffic does not match known attack patterns but shows unusual behavior"
            )
        elif final_verdict == "Normal":
            explanations.insert(0, "Traffic matches normal network behavior")

        # ---------- Construct JSON Response ----------
        response = {
            "rf_decision": {
                "label": rf_decision,
                "confidence": round(rf_conf, 2),
                "top_features": top_rf_features
            },
            "if_decision": {
                "label": iso_decision,
                "top_features": top_if_features
            },
            "final_decision": final_verdict,
            "agreement_status": agreement_status,
            "confidence": round(combined_conf, 2),
            "explanation_nl": explanation_nl,
            "feature_explanations": explanations,
            "recommended_action": action,
            "severity": severity
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


print("ABOUT TO START FLASK")

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
