from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import sys

# ── Fix: tell Python where to find the 'models' module ────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)  # adds churn_predictor/ to Python's search path

app = Flask(__name__)

# ── Load everything once at startup ───────────────────────────────────
with open(os.path.join(BASE, 'models/trained_model.pkl'),   'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE, 'models/best_threshold.pkl'),  'rb') as f:
    best_threshold = pickle.load(f)
with open(os.path.join(BASE, 'data/scaler.pkl'),            'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE, 'data/feature_cols.pkl'),      'rb') as f:
    feature_cols = pickle.load(f)
with open(os.path.join(BASE, 'data/ordinal_maps.pkl'),      'rb') as f:
    ordinal_maps = pickle.load(f)

print(f"[OK] Model loaded | Threshold: {best_threshold:.2f}")

# ── Analytics data (pre-computed from training) ────────────────────────
ANALYTICS = {
    'accuracy'  : 84.5,
    'precision' : 67.8,
    'recall'    : 76.9,
    'f1'        : 72.1,
    'auc'       : 88.6,
    'confusion' : {'tn': 125, 'fp': 23, 'fn': 8, 'tp': 44},
    'f1_comparison': {
        'Logistic Reg': 62,
        'Random Forest': 71,
        'XGBoost'      : 74,
        'Our Model'    : 72
    }
}

# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', data=ANALYTICS)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives form data, preprocesses it exactly like training,
    runs the model, returns prediction + explanation.
    """
    try:
        # ── Read form values ───────────────────────────────────────────
        usage_duration    = request.form.get('usage_duration')
        subscription_plan = request.form.get('subscription_plan')
        daily_activity    = int(request.form.get('daily_activity', 20))
        session_length    = request.form.get('session_length')
        usage_frequency   = request.form.get('usage_frequency')
        experience_rating = int(request.form.get('experience_rating', 3))
        last_active       = request.form.get('last_active')

        # ── Encode exactly as training ─────────────────────────────────
        encoded = [
            ordinal_maps['usage_duration'][usage_duration],
            ordinal_maps['subscription_plan'][subscription_plan],
            daily_activity,
            ordinal_maps['session_length'][session_length],
            ordinal_maps['usage_frequency'][usage_frequency],
            experience_rating,
            ordinal_maps['last_active'][last_active],
        ]

        # ── Scale using saved scaler ───────────────────────────────────
        X = np.array(encoded).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # ── Predict ────────────────────────────────────────────────────
        churn_prob   = float(model.predict_proba(X_scaled)[0])
        churn_pred   = int(churn_prob >= best_threshold)
        churn_pct    = round(churn_prob * 100, 1)

        # ── Build explanation based on input values ────────────────────
        explanations = []
        risk_factors = []

        if daily_activity > 40:
            explanations.append("High engagement in daily activity")
        elif daily_activity < 10:
            risk_factors.append("Very low daily message activity")

        if usage_frequency == 'High (5-7)':
            explanations.append("Regular usage frequency")
        elif usage_frequency == 'Low (1-2)':
            risk_factors.append("Low usage frequency (1-2 days/week)")

        if last_active in ['Today', '1-3 days']:
            explanations.append("Recent active status")
        elif last_active in ['8-14 days', '15+ days']:
            risk_factors.append(f"Last active: {last_active} ago")

        if experience_rating >= 4:
            explanations.append("Good overall experience rating")
        elif experience_rating <= 2:
            risk_factors.append(f"Low experience rating ({experience_rating} stars)")

        if subscription_plan in ['Pro', 'Enterprise']:
            explanations.append(f"{subscription_plan} plan user")
        elif subscription_plan == 'Free':
            risk_factors.append("Free plan — no paid commitment")

        # Determine risk level
        if churn_pct < 30:
            risk_level = "Low Risk"
        elif churn_pct < 60:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        result = {
            'churn_pred'   : churn_pred,
            'churn_pct'    : churn_pct,
            'risk_level'   : risk_level,
            'explanations' : explanations if not churn_pred else risk_factors,
            'label'        : 'CHURN' if churn_pred else 'STAY',
            # Echo back inputs for result page display
            'inputs': {
                'Usage Duration'   : usage_duration,
                'Plan'             : subscription_plan,
                'Daily Activity'   : f'{daily_activity} msgs/day',
                'Session Length'   : session_length,
                'Usage Frequency'  : usage_frequency,
                'Experience Rating': f'{experience_rating} stars',
                'Last Active'      : last_active,
            }
        }
        return render_template('result.html', result=result)

    except Exception as e:
        return f"<h3>Error: {e}</h3><a href='/predict'>Go back</a>", 400


if __name__ == '__main__':
    app.run(debug=True)