import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

def preprocess(filepath='data/chatbot_churn.csv', save=True):

    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    print(f"Churn rate before SMOTE: {df['churn'].mean():.1%}")

    # ── Step A: Ordinal Encoding ───────────────────────────────────────────
    # These features have a clear ORDER — so we use numbers that preserve rank.
    # Example: 'Today' = 0 (most active), '15+ days' = 4 (least active)

    ordinal_maps = {
        'usage_duration': {
            '0-3 months': 0, '3-6 months': 1,
            '6-12 months': 2, '12+ months': 3
        },
        'session_length': {
            '<5 min': 0, '5-15 min': 1,
            '15-30 min': 2, '30+ min': 3
        },
        'usage_frequency': {
            'Low (1-2)': 0, 'Medium (3-4)': 1, 'High (5-7)': 2
        },
        'subscription_plan': {
            'Free': 0, 'Basic': 1, 'Pro': 2, 'Enterprise': 3
        },
        'last_active': {
            'Today': 0, '1-3 days': 1, '4-7 days': 2,
            '8-14 days': 3, '15+ days': 4
        }
    }

    for col, mapping in ordinal_maps.items():
        df[col] = df[col].map(mapping)

    print("\nAfter encoding — sample:")
    print(df.head(3))

    # ── Step B: Feature Matrix & Target ───────────────────────────────────
    feature_cols = [
        'usage_duration', 'subscription_plan', 'daily_activity',
        'session_length', 'usage_frequency', 'experience_rating', 'last_active'
    ]
    X = df[feature_cols].values   # shape: (1000, 7)
    y = df['churn'].values         # shape: (1000,)

    # ── Step C: Train / Test Split ─────────────────────────────────────────
    # 80% train, 20% test | stratify=y → keeps churn % same in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # ── Step D: Feature Scaling ────────────────────────────────────────────
    # WHY scale? daily_activity goes 0-100 but usage_duration goes 0-3.
    # Without scaling, gradient descent overweights high-range features.
    # StandardScaler: z = (x - mean) / std → mean=0, std=1 for each feature

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit on train only!
    X_test_scaled  = scaler.transform(X_test)        # apply same transform to test

    print(f"\nAfter scaling — feature means (should be ~0): {X_train_scaled.mean(axis=0).round(2)}")
    print(f"After scaling — feature stds  (should be ~1): {X_train_scaled.std(axis=0).round(2)}")

    # ── Step E: SMOTE (Synthetic Minority Oversampling) ────────────────────
    # WHY? We have 738 "stay" vs 262 "churn". A lazy model that always says
    # "stay" would be 73.8% accurate — that's useless. SMOTE creates synthetic
    # churn samples by interpolating between real churn examples.
    # CRITICAL: Apply SMOTE only on TRAINING data, never on test data!

    print(f"\nBefore SMOTE — Train class counts: {np.bincount(y_train)}")

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    print(f"After  SMOTE — Train class counts: {np.bincount(y_train_bal)}")
    print(f"New train size: {X_train_bal.shape[0]}")

    # ── Step F: Save everything ────────────────────────────────────────────
    if save:
        np.save('data/X_train.npy', X_train_bal)
        np.save('data/X_test.npy',  X_test_scaled)
        np.save('data/y_train.npy', y_train_bal)
        np.save('data/y_test.npy',  y_test)
        with open('data/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('data/feature_cols.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        with open('data/ordinal_maps.pkl', 'wb') as f:
            pickle.dump(ordinal_maps, f)
        print("\n[Saved] X_train, X_test, y_train, y_test, scaler, feature_cols, ordinal_maps")

    return X_train_bal, X_test_scaled, y_train_bal, y_test, scaler, feature_cols

if __name__ == '__main__':
    preprocess()