import numpy as np

class LogisticRegressionScratch:
    """
    Logistic Regression built from scratch using only NumPy.
    
    WHAT THIS CLASS DOES, STEP BY STEP:
    ─────────────────────────────────────
    1. Takes input features X (shape: n_samples x n_features)
    2. Computes a weighted sum: z = X·weights + bias
    3. Squashes z to 0-1 range using sigmoid: p = 1 / (1 + e^(-z))
    4. Compares p to true labels using log-loss (cross-entropy)
    5. Computes gradients (how much each weight contributed to error)
    6. Updates weights to reduce the loss — repeated for n_iterations
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        learning_rate : how big a step we take each iteration
                        too high → overshoots minimum
                        too low  → takes forever to converge
        n_iterations  : how many times we update the weights
        """
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.weights       = None   # shape: (n_features,)
        self.bias          = None   # single scalar
        self.loss_history  = []     # track loss per iteration (for plotting)

    # ──────────────────────────────────────────────────────────────────────
    # CORE MATH #1 — Sigmoid Function
    # ──────────────────────────────────────────────────────────────────────
    def sigmoid(self, z):
        """
        Converts any real number into a probability between 0 and 1.
        
        Formula: σ(z) = 1 / (1 + e^(-z))
        
        Examples:
          sigmoid(0)    = 0.5   (completely uncertain)
          sigmoid(5)    = 0.993 (very likely churn)
          sigmoid(-5)   = 0.007 (very likely stay)
        
        WHY CLIPPING? When z is very large/small (e.g. z = -800),
        np.exp(-z) becomes inf → division by inf → NaN.
        Clipping z to [-500, 500] prevents this overflow safely.
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # ──────────────────────────────────────────────────────────────────────
    # CORE MATH #2 — Log Loss (Binary Cross-Entropy)
    # ──────────────────────────────────────────────────────────────────────
    def compute_loss(self, y_true, y_pred):
        """
        Measures HOW WRONG the model currently is.
        
        Formula:
          L = -1/n * Σ [ y·log(p) + (1-y)·log(1-p) ]
        
        INTUITION:
          - If true label y=1 (churn) and model says p=0.95 → small loss ✓
          - If true label y=1 (churn) and model says p=0.05 → huge loss ✗
          - If true label y=0 (stay)  and model says p=0.05 → small loss ✓
          - If true label y=0 (stay)  and model says p=0.95 → huge loss ✗
        
        WHY LOG? Because log(p) → -∞ as p → 0. If you're very confident
        and very wrong, you get punished severely. This forces the model
        to be calibrated, not overconfident.
        
        WHY EPSILON? log(0) = -∞ which breaks math. We clip predictions
        to a tiny range away from 0 and 1 to keep numbers stable.
        """
        n = len(y_true)
        epsilon = 1e-15   # prevents log(0)
        y_pred  = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -1/n * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    # ──────────────────────────────────────────────────────────────────────
    # CORE MATH #3 — Gradient Descent
    # ──────────────────────────────────────────────────────────────────────
    def fit(self, X, y):
        """
        TRAINING LOOP — where the model actually learns.
        
        GRADIENT DESCENT INTUITION:
        Imagine the loss as a bowl-shaped valley. Your weights start at a
        random point on the rim. Each iteration you look at the slope
        (gradient) and take one step downhill. After enough steps you
        reach the bottom = minimum loss = best weights.
        
        GRADIENT FORMULAS (derived from calculus on the log-loss):
          dL/dw = (1/n) · Xᵀ · (predictions - true_labels)
          dL/db = (1/n) · Σ(predictions - true_labels)
        
        UPDATE RULE:
          weights = weights - learning_rate × dL/dw
          bias    = bias    - learning_rate × dL/db
        
        WHY SUBTRACT? Because gradient points uphill. We want downhill.
        """
        n_samples, n_features = X.shape

        # ── Step 1: Initialize weights to zero ────────────────────────────
        # Starting at zero is fine for logistic regression (unlike neural nets)
        self.weights = np.zeros(n_features)
        self.bias    = 0.0

        print(f"\n{'='*55}")
        print(f"  TRAINING — Logistic Regression from Scratch")
        print(f"{'='*55}")
        print(f"  Samples     : {n_samples}")
        print(f"  Features    : {n_features}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations  : {self.n_iterations}")
        print(f"{'='*55}")

        # ── Step 2: Gradient descent loop ─────────────────────────────────
        for i in range(self.n_iterations):

            # Forward pass: compute predictions
            # z = X·w + b  →  shape: (n_samples,)
            z = np.dot(X, self.weights) + self.bias

            # Apply sigmoid to get probabilities 0-1
            predictions = self.sigmoid(z)

            # Compute loss for this iteration
            loss = self.compute_loss(y, predictions)
            self.loss_history.append(loss)

            # Compute gradients
            # error = predictions - true labels  →  shape: (n_samples,)
            error = predictions - y

            # dL/dw: how much each weight contributed to the error
            # X.T shape: (n_features, n_samples)
            # error shape: (n_samples,)
            # result: (n_features,) — one gradient per weight
            dw = (1 / n_samples) * np.dot(X.T, error)

            # dL/db: gradient for the bias term
            db = (1 / n_samples) * np.sum(error)

            # Update weights (move downhill)
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i+1:5d} | Loss: {loss:.6f} | "
                      f"Weights range: [{self.weights.min():.3f}, {self.weights.max():.3f}]")

        print(f"{'='*55}")
        print(f"  Training complete!")
        print(f"  Final loss : {self.loss_history[-1]:.6f}")
        print(f"  Loss drop  : {self.loss_history[0]:.4f} → {self.loss_history[-1]:.4f}")
        print(f"{'='*55}\n")

        return self

    # ──────────────────────────────────────────────────────────────────────
    # PREDICTION
    # ──────────────────────────────────────────────────────────────────────
    def predict_proba(self, X):
        """
        Returns raw probability of churn for each sample.
        Example output: [0.12, 0.87, 0.34, 0.91, ...]
        This is what the web app will show as "Churn Probability %"
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Converts probabilities to binary predictions (0=stay, 1=churn).
        
        threshold=0.5 means:
          prob >= 0.5 → predict Churn  (1)
          prob <  0.5 → predict Stay   (0)
        
        You can lower threshold (e.g. 0.3) to catch MORE churners
        at the cost of more false alarms. This is the Precision/Recall
        tradeoff you will see in Phase 4.
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, feature_names):
        """
        The learned weights tell us which features matter most.
        Large positive weight = strongly predicts CHURN
        Large negative weight = strongly predicts STAY
        """
        importance = list(zip(feature_names, self.weights))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)
        return importance