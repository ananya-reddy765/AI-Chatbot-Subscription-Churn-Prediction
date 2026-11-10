import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.logistic_regression import LogisticRegressionScratch

# ── Load preprocessed data ─────────────────────────────────────────────
X_train = np.load('data/X_train.npy')
X_test  = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test  = np.load('data/y_test.npy')

with open('data/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"Features: {feature_cols}")

# ══════════════════════════════════════════════════════════════════════════
# TRAIN YOUR MODEL FROM SCRATCH
# ══════════════════════════════════════════════════════════════════════════
model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# Save trained model
with open('models/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("[Saved] models/trained_model.pkl")

# ══════════════════════════════════════════════════════════════════════════
# EVALUATE ON TEST SET
# ══════════════════════════════════════════════════════════════════════════
y_pred      = model.predict(X_test)           # binary: 0 or 1
y_pred_prob = model.predict_proba(X_test)     # float:  0.0 to 1.0

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
cm        = confusion_matrix(y_test, y_pred)

print("\n" + "="*55)
print("  EVALUATION RESULTS — Custom Model (from scratch)")
print("="*55)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print("="*55)
print(f"\n  Confusion Matrix:")
print(f"  ┌─────────────────────────────┐")
print(f"  │           Predicted         │")
print(f"  │        Stay    Churn        │")
print(f"  │ Stay  [ {cm[0][0]:4d}    {cm[0][1]:4d} ]       │")
print(f"  │ Churn [ {cm[1][0]:4d}    {cm[1][1]:4d} ]       │")
print(f"  └─────────────────────────────┘")
print(f"  True Negatives  (Stay→Stay)  : {cm[0][0]}")
print(f"  False Positives (Stay→Churn) : {cm[0][1]}")
print(f"  False Negatives (Churn→Stay) : {cm[1][0]}")
print(f"  True Positives  (Churn→Churn): {cm[1][1]}")

# Feature importance
print(f"\n  Feature Importance (by learned weight):")
importance = model.get_feature_importance(feature_cols)
for feat, weight in importance:
    direction = "→ churn" if weight > 0 else "→ stay"
    bar = "█" * int(abs(weight) * 10)
    print(f"  {feat:22s}: {weight:+.4f}  {direction}  {bar}")

# ══════════════════════════════════════════════════════════════════════════
# VISUALIZE — 4-panel results dashboard
# ══════════════════════════════════════════════════════════════════════════
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Model Training Results — Custom Logistic Regression',
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

# ── Plot 1: Loss curve ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(model.loss_history, color='#2563EB', linewidth=2)
ax1.fill_between(range(len(model.loss_history)), model.loss_history,
                 alpha=0.15, color='#2563EB')
ax1.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Log Loss', fontsize=11)
ax1.annotate(f'Start: {model.loss_history[0]:.3f}',
             xy=(0, model.loss_history[0]), xytext=(100, model.loss_history[0] + 0.05),
             fontsize=10, color='#EF4444',
             arrowprops=dict(arrowstyle='->', color='#EF4444'))
ax1.annotate(f'End: {model.loss_history[-1]:.3f}',
             xy=(len(model.loss_history)-1, model.loss_history[-1]),
             xytext=(len(model.loss_history)-300, model.loss_history[-1] + 0.05),
             fontsize=10, color='#10B981',
             arrowprops=dict(arrowstyle='->', color='#10B981'))

# ── Plot 2: Confusion Matrix heatmap ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
ax2.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Stay (0)', 'Churn (1)'], fontsize=11)
ax2.set_yticklabels(['Stay (0)', 'Churn (1)'], fontsize=11)
ax2.set_xlabel('Predicted', fontsize=11)
ax2.set_ylabel('Actual', fontsize=11)
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{cm[i,j]}',
                 ha='center', va='center', fontsize=18, fontweight='bold',
                 color='white' if cm[i,j] > thresh else 'black')
plt.colorbar(im, ax=ax2)

# ── Plot 3: ROC Curve ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
ax3.plot(fpr, tpr, color='#8B5CF6', linewidth=2.5,
         label=f'Custom Model (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')
ax3.fill_between(fpr, tpr, alpha=0.15, color='#8B5CF6')
ax3.set_title('ROC Curve', fontsize=13, fontweight='bold')
ax3.set_xlabel('False Positive Rate', fontsize=11)
ax3.set_ylabel('True Positive Rate', fontsize=11)
ax3.legend(fontsize=10)
ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1])

# ── Plot 4: Feature Importance ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
feat_names = [f for f, _ in importance]
feat_weights = [w for _, w in importance]
colors = ['#EF4444' if w > 0 else '#10B981' for w in feat_weights]
bars = ax4.barh(feat_names, feat_weights, color=colors, edgecolor='white', linewidth=1.2)
ax4.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
ax4.set_title('Feature Importance\n(red=churn, green=stay)',
              fontsize=13, fontweight='bold')
ax4.set_xlabel('Learned Weight', fontsize=11)
for bar, val in zip(bars, feat_weights):
    ax4.text(val + (0.005 if val >= 0 else -0.005),
             bar.get_y() + bar.get_height()/2,
             f'{val:+.3f}', va='center',
             ha='left' if val >= 0 else 'right', fontsize=9)

# ── Metrics text box ──────────────────────────────────────────────────
metrics_text = (f"Accuracy:  {accuracy*100:.1f}%\n"
                f"Precision: {precision*100:.1f}%\n"
                f"Recall:    {recall*100:.1f}%\n"
                f"F1 Score:  {f1*100:.1f}%\n"
                f"ROC-AUC:   {roc_auc:.3f}")
ax3.text(0.62, 0.18, metrics_text, transform=ax3.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig('data/training_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[Saved] data/training_results.png")