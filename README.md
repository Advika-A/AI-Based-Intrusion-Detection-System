AI-Based Intrusion Detection System (AI-IDS)
📌 Overview

This project implements a hybrid AI-based Intrusion Detection System (IDS) that combines supervised learning and unsupervised anomaly detection to identify both known and unknown network attacks.
The system not only detects intrusions but also provides explainable insights at the feature level, making it suitable for educational demos and real-world monitoring.

Key highlights:

Detects known attacks using a Random Forest classifier.

Detects zero-day or unusual traffic using Isolation Forest.

Provides feature-level explanations to help understand why a traffic sample is suspicious.

Simulated demo mode supports Normal, Suspicious, and Attack traffic with visual indicators.

🚀 Key Features

Hybrid Detection: Combines supervised Random Forest and unsupervised Isolation Forest for comprehensive coverage.

Explainable AI (XAI): Shows top features contributing to the model’s decision and natural language explanations.

Real-Time Frontend: Interactive web interface for testing traffic scenarios (Normal, Suspicious, Attack).

Confidence Scoring: Displays overall confidence and agreement between models.

Action Recommendations: Suggests actions based on detected severity.

🧠 Datasets Used

NSL-KDD: Baseline intrusion detection dataset for model training.

CICIDS2017: (Planned) For modern attack patterns and anomaly detection.

🛠️ Technologies

Backend: Python, Flask, Scikit-learn, Pandas, NumPy

Frontend: HTML, CSS, JavaScript (interactive dashboard)

Version Control: Git & GitHub

📂 Project Structure
AI-IDS/
├── data/                     # Raw and processed datasets
├── models/                   # Trained ML models (Random Forest & Isolation Forest)
├── src/                      # Python scripts and API code
│   └── api.py                # Flask API serving predictions
├── frontend/                 # HTML, CSS, JS frontend files
├── docs/                     # Documentation & diagrams
├── README.md                 # Project overview
└── requirements.txt          # Python dependencies

🔧 How to Run the Project

Clone the repository

git clone https://github.com/yourusername/AI-IDS.git
cd AI-IDS


Install dependencies

pip install -r requirements.txt


Start the backend server

python src/api.py


Open the frontend
Open frontend/index_ids.html in a web browser and test Normal, Suspicious, or Attack traffic scenarios.

Check the console
The Flask backend logs all requests and predictions for easy debugging.

📊 Explainable AI Features

Top Features: Each model highlights 3–4 most important features influencing its decision.

Deviation Scores: Visual bars show how extreme feature values are compared to normal traffic.

Natural Language Explanations: The system explains why a traffic flow is suspicious in plain English.

Final Verdict: Combines supervised and unsupervised outputs to classify traffic as Normal, Suspicious, or Attack.

📈 Future Work

CICIDS2017 Integration: Train models on modern attack datasets.

FastAPI Backend: For faster, production-ready, real-time detection.

Advanced Frontend Dashboard: Graphs, charts, and logs for network monitoring.

Model Optimization: Reduce false positives and improve anomaly detection accuracy.

Explainability Enhancements: Use SHAP/LIME for precise feature attribution.
