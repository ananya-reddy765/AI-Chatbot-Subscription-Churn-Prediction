import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
import pickle, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.logistic_regression import LogisticRegressionScratch

# ── Load data & models ─────────────────────────────────────────────────
X_train = np.load('data/X_train.npy')
X_test  = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test  = np.load('data/y_test.npy')

with open('data/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
with open('models/trained_model.pkl', 'rb') as f:
    custom_model = pickle.load(f)

# ══════════════════════════════════════════════════════════════════════════
# TRAIN SKLEARN MODEL (benchmark)
# ══════════════════════════════════════════════════════════════════════════
sklearn_model = LogisticRegression(max_iter=1000, random_state=42)
sklearn_model.fit(X_train, y_train)

# ── Get predictions from both models ──────────────────────────────────
custom_prob  = custom_model.predict_proba(X_test)
sklearn_prob = sklearn_model.predict_proba(X_test)[:, 1]

custom_pred  = custom_model.predict(X_test)
sklearn_pred = sklearn_model.predict(X_test)

# ══════════════════════════════════════════════════════════════════════════
# METRICS COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════
def get_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy' : accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall'   : recall_score(y_true, y_pred),
        'F1'       : f1_score(y_true, y_pred),
        'ROC-AUC'  : roc_auc_score(y_true, y_prob),
    }

custom_metrics  = get_metrics(y_test, custom_pred,  custom_prob)
sklearn_metrics = get_metrics(y_test, sklearn_pred, sklearn_prob)

print("\n" + "="*62)
print("  BENCHMARK — Custom (Scratch) vs Sklearn LogisticRegression")
print("="*62)
print(f"  {'Metric':<12}  {'Custom':>10}  {'Sklearn':>10}  {'Diff':>10}")
print(f"  {'-'*50}")
for metric in custom_metrics:
    c = custom_metrics[metric]
    s = sklearn_metrics[metric]
    diff = c - s
    symbol = "+" if diff >= 0 else ""
    print(f"  {metric:<12}  {c:>10.4f}  {s:>10.4f}  {symbol+f'{diff:.4f}':>10}")
print("="*62)

# ══════════════════════════════════════════════════════════════════════════
# THRESHOLD TUNING — find best threshold for F1
# ══════════════════════════════════════════════════════════════════════════
print("\n  THRESHOLD ANALYSIS (Custom Model):")
print(f"  {'Threshold':>10}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
print(f"  {'-'*55}")

thresholds   = np.arange(0.2, 0.81, 0.05)
best_f1      = 0
best_thresh  = 0.5
thresh_results = []

for t in thresholds:
    preds = (custom_prob >= t).astype(int)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    thresh_results.append((t, acc, prec, rec, f1))
    marker = " ← best F1" if f1 > best_f1 else ""
    if f1 > best_f1:
        best_f1     = f1
        best_thresh = t
    print(f"  {t:>10.2f}  {acc:>10.4f}  {prec:>10.4f}  {rec:>10.4f}  {f1:>10.4f}{marker}")

print(f"\n  Best threshold: {best_thresh:.2f} | Best F1: {best_f1:.4f}")

# Save best threshold
with open('models/best_threshold.pkl', 'wb') as f:
    pickle.dump(best_thresh, f)
print(f"  [Saved] models/best_threshold.pkl")

# ══════════════════════════════════════════════════════════════════════════
# VISUALIZE — 6-panel final dashboard
# ══════════════════════════════════════════════════════════════════════════
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Phase 4 — Full Evaluation & Sklearn Benchmark',
             fontsize=17, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

BLUE   = '#2563EB'
GREEN  = '#10B981'
PURPLE = '#8B5CF6'
RED    = '#EF4444'
ORANGE = '#F97316'

# ── Plot 1: Metrics bar chart comparison ─────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
metric_names = list(custom_metrics.keys())
x = np.arange(len(metric_names))
w = 0.35
bars1 = ax1.bar(x - w/2, [custom_metrics[m]  for m in metric_names],
                w, label='Custom (Scratch)', color=BLUE,   alpha=0.85, edgecolor='white')
bars2 = ax1.bar(x + w/2, [sklearn_metrics[m] for m in metric_names],
                w, label='Sklearn',          color=ORANGE, alpha=0.85, edgecolor='white')
ax1.set_title('Custom vs Sklearn — All Metrics', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metric_names, rotation=15, fontsize=10)
ax1.set_ylim(0, 1.1)
ax1.legend(fontsize=10)
ax1.set_ylabel('Score', fontsize=11)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.2f}', ha='center', fontsize=8, color=BLUE, fontweight='bold')
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.2f}', ha='center', fontsize=8, color=ORANGE, fontweight='bold')

# ── Plot 2: ROC Curves — both models ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
fpr_c, tpr_c, _ = roc_curve(y_test, custom_prob)
fpr_s, tpr_s, _ = roc_curve(y_test, sklearn_prob)
ax2.plot(fpr_c, tpr_c, color=BLUE,   lw=2.5,
         label=f'Custom  (AUC={custom_metrics["ROC-AUC"]:.3f})')
ax2.plot(fpr_s, tpr_s, color=ORANGE, lw=2.5, linestyle='--',
         label=f'Sklearn (AUC={sklearn_metrics["ROC-AUC"]:.3f})')
ax2.plot([0,1],[0,1],'k--', lw=1, alpha=0.4, label='Random (0.500)')
ax2.fill_between(fpr_c, tpr_c, alpha=0.10, color=BLUE)
ax2.set_title('ROC Curve Comparison', fontsize=13, fontweight='bold')
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.legend(fontsize=10)
ax2.set_xlim([0,1]); ax2.set_ylim([0,1])

# ── Plot 3: Precision-Recall curve ───────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
prec_c, rec_c, _ = precision_recall_curve(y_test, custom_prob)
prec_s, rec_s, _ = precision_recall_curve(y_test, sklearn_prob)
ax3.plot(rec_c, prec_c, color=BLUE,   lw=2.5, label='Custom')
ax3.plot(rec_s, prec_s, color=ORANGE, lw=2.5, linestyle='--', label='Sklearn')
baseline = y_test.mean()
ax3.axhline(y=baseline, color='gray', linestyle=':', lw=1.5,
            label=f'Baseline ({baseline:.2f})')
ax3.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
ax3.set_xlabel('Recall', fontsize=11)
ax3.set_ylabel('Precision', fontsize=11)
ax3.legend(fontsize=10)
ax3.set_xlim([0,1]); ax3.set_ylim([0,1])

# ── Plot 4: Threshold tuning chart ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
t_vals  = [r[0] for r in thresh_results]
acc_v   = [r[1] for r in thresh_results]
prec_v  = [r[2] for r in thresh_results]
rec_v   = [r[3] for r in thresh_results]
f1_v    = [r[4] for r in thresh_results]
ax4.plot(t_vals, acc_v,  marker='o', color=BLUE,   lw=2, label='Accuracy')
ax4.plot(t_vals, prec_v, marker='s', color=RED,    lw=2, label='Precision')
ax4.plot(t_vals, rec_v,  marker='^', color=GREEN,  lw=2, label='Recall')
ax4.plot(t_vals, f1_v,   marker='D', color=PURPLE, lw=2.5, label='F1', zorder=5)
ax4.axvline(x=best_thresh, color='black', linestyle='--', lw=1.5,
            label=f'Best threshold={best_thresh:.2f}')
ax4.set_title('Threshold vs Metrics', fontsize=13, fontweight='bold')
ax4.set_xlabel('Decision Threshold', fontsize=11)
ax4.set_ylabel('Score', fontsize=11)
ax4.legend(fontsize=9)
ax4.set_xlim([0.18, 0.82]); ax4.set_ylim([0, 1.05])

# ── Plot 5: Confusion matrices side by side ──────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
cm_custom  = confusion_matrix(y_test, custom_pred)
cm_sklearn = confusion_matrix(y_test, sklearn_pred)
cm_combined = np.zeros((4, 2))  # stack vertically for display

# Show both confusion matrices as annotated heatmap pairs
for idx, (cm_plot, title, ax_use) in enumerate([
    (cm_custom,  'Confusion Matrix — Custom',  ax5),
    (cm_sklearn, 'Confusion Matrix — Sklearn', fig.add_subplot(gs[1, 2]))
]):
    im = ax_use.imshow(cm_plot, cmap='Blues', interpolation='nearest')
    ax_use.set_title(title, fontsize=13, fontweight='bold')
    ax_use.set_xticks([0,1]); ax_use.set_yticks([0,1])
    ax_use.set_xticklabels(['Stay','Churn'], fontsize=11)
    ax_use.set_yticklabels(['Stay','Churn'], fontsize=11)
    ax_use.set_xlabel('Predicted', fontsize=11)
    ax_use.set_ylabel('Actual', fontsize=11)
    thresh_cm = cm_plot.max() / 2
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            ax_use.text(j, i, f'{labels[i][j]}\n{cm_plot[i,j]}',
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        color='white' if cm_plot[i,j] > thresh_cm else 'black')

plt.savefig('data/evaluation_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[Saved] data/evaluation_dashboard.png")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  FINAL SUMMARY")
print("="*62)
print(f"  Your custom model ROC-AUC : {custom_metrics['ROC-AUC']:.4f}")
print(f"  Sklearn  model ROC-AUC    : {sklearn_metrics['ROC-AUC']:.4f}")
gap = abs(custom_metrics['ROC-AUC'] - sklearn_metrics['ROC-AUC'])
print(f"  Gap (AUC difference)      : {gap:.4f}")
if gap < 0.03:
    print("  Verdict: Your scratch model matches sklearn — excellent!")
elif gap < 0.06:
    print("  Verdict: Very close to sklearn — strong implementation.")
else:
    print("  Verdict: Some gap — expected, sklearn uses more optimizations.")
print(f"\n  Best decision threshold   : {best_thresh:.2f}")
print(f"  Best F1 at that threshold : {best_f1:.4f}")
print("="*62)