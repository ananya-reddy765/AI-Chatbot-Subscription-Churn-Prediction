import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

# --- Feature generation ---
# How long they've been subscribed (months)
usage_duration = np.random.choice(
    ['0-3 months', '3-6 months', '6-12 months', '12+ months'],
    size=n, p=[0.25, 0.25, 0.25, 0.25]
)

# Subscription plan
subscription_plan = np.random.choice(
    ['Free', 'Basic', 'Pro', 'Enterprise'],
    size=n, p=[0.35, 0.30, 0.25, 0.10]
)

# Daily messages sent (0-100)
daily_activity = np.random.randint(0, 101, size=n)

# Session length in minutes
session_length = np.random.choice(
    ['<5 min', '5-15 min', '15-30 min', '30+ min'],
    size=n, p=[0.20, 0.30, 0.30, 0.20]
)

# Usage frequency (active days per week)
usage_frequency = np.random.choice(
    ['Low (1-2)', 'Medium (3-4)', 'High (5-7)'],
    size=n, p=[0.30, 0.40, 0.30]
)

# User experience rating (1-5 stars)
experience_rating = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.10, 0.15, 0.25, 0.30, 0.20])

# Last active (days ago)
last_active = np.random.choice(
    ['Today', '1-3 days', '4-7 days', '8-14 days', '15+ days'],
    size=n, p=[0.25, 0.20, 0.20, 0.15, 0.20]
)

# --- Churn logic (rule-based for realism) ---
churn_score = np.zeros(n)

# Low activity → more churn
churn_score += np.where(daily_activity < 10, 1.5, 0)
churn_score += np.where(daily_activity > 40, -1.0, 0)

# Free plan → more churn
churn_score += np.where(subscription_plan == 'Free', 1.2, 0)
churn_score += np.where(subscription_plan == 'Pro', -0.8, 0)
churn_score += np.where(subscription_plan == 'Enterprise', -1.2, 0)

# Low rating → churn
churn_score += np.where(experience_rating <= 2, 1.5, 0)
churn_score += np.where(experience_rating >= 4, -1.0, 0)

# Inactive recently → churn
churn_score += np.where(last_active == '15+ days', 2.0, 0)
churn_score += np.where(last_active == 'Today', -1.5, 0)

# Short session → churn
churn_score += np.where(session_length == '<5 min', 0.8, 0)

# Low frequency → churn
churn_score += np.where(usage_frequency == 'Low (1-2)', 1.0, 0)
churn_score += np.where(usage_frequency == 'High (5-7)', -0.8, 0)

# Convert score to probability using sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

churn_prob = sigmoid(churn_score - 1.5)  # offset to get ~25% churn rate
churn = (np.random.rand(n) < churn_prob).astype(int)

# --- Build DataFrame ---
df = pd.DataFrame({
    'usage_duration': usage_duration,
    'subscription_plan': subscription_plan,
    'daily_activity': daily_activity,
    'session_length': session_length,
    'usage_frequency': usage_frequency,
    'experience_rating': experience_rating,
    'last_active': last_active,
    'churn': churn
})

df.to_csv('data/chatbot_churn.csv', index=False)
print(f"Dataset saved! Shape: {df.shape}")
print(f"\nChurn distribution:\n{df['churn'].value_counts()}")
print(f"\nChurn rate: {df['churn'].mean():.1%}")
print(f"\nFirst 5 rows:\n{df.head()}")