import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/chatbot_churn.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape        : {df.shape}")
print(f"Churn rate   : {df['churn'].mean():.1%}")
print(f"Missing vals :\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nNumerical summary:\n{df.describe()}")

# ── Style ──────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'stay': '#2563EB', 'churn': '#EF4444'}
CAT_PALETTE = ['#2563EB', '#EF4444']

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Overview dashboard (2x3 grid)
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Chatbot Subscription — Churn EDA Dashboard', fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Overall churn distribution (pie) ────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
counts = df['churn'].value_counts()
wedges, texts, autotexts = ax1.pie(
    counts, labels=['Stay', 'Churn'],
    autopct='%1.1f%%', colors=[COLORS['stay'], COLORS['churn']],
    startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for at in autotexts:
    at.set_fontsize(12); at.set_fontweight('bold'); at.set_color('white')
ax1.set_title('Overall Churn Distribution', fontsize=13, fontweight='bold', pad=12)

# ── Plot 2: Churn by Subscription Plan ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
plan_order = ['Free', 'Basic', 'Pro', 'Enterprise']
plan_churn = df.groupby('subscription_plan')['churn'].mean() * 100
plan_churn = plan_churn.reindex(plan_order)
bars = ax2.bar(plan_churn.index, plan_churn.values,
               color=['#EF4444', '#F97316', '#3B82F6', '#10B981'], edgecolor='white', linewidth=1.5)
ax2.set_title('Churn Rate by Subscription Plan', fontsize=13, fontweight='bold')
ax2.set_ylabel('Churn Rate (%)', fontsize=11)
ax2.set_ylim(0, 100)
for bar, val in zip(bars, plan_churn.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# ── Plot 3: Churn by Experience Rating ──────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
rating_churn = df.groupby('experience_rating')['churn'].mean() * 100
ax3.plot(rating_churn.index, rating_churn.values,
         marker='o', color='#8B5CF6', linewidth=2.5, markersize=9, markerfacecolor='white',
         markeredgewidth=2.5)
ax3.fill_between(rating_churn.index, rating_churn.values, alpha=0.15, color='#8B5CF6')
ax3.set_title('Churn Rate by Experience Rating', fontsize=13, fontweight='bold')
ax3.set_xlabel('Star Rating (1–5)', fontsize=11)
ax3.set_ylabel('Churn Rate (%)', fontsize=11)
ax3.set_xticks([1, 2, 3, 4, 5])
for x, y in zip(rating_churn.index, rating_churn.values):
    ax3.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')

# ── Plot 4: Daily Activity distribution (churn vs stay) ─────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df[df['churn']==0]['daily_activity'], bins=20, alpha=0.7,
         color=COLORS['stay'], label='Stay', edgecolor='white')
ax4.hist(df[df['churn']==1]['daily_activity'], bins=20, alpha=0.7,
         color=COLORS['churn'], label='Churn', edgecolor='white')
ax4.set_title('Daily Activity: Stay vs Churn', fontsize=13, fontweight='bold')
ax4.set_xlabel('Daily Messages Sent', fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.legend(fontsize=11)

# ── Plot 5: Churn by Last Active ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
active_order = ['Today', '1-3 days', '4-7 days', '8-14 days', '15+ days']
active_churn = df.groupby('last_active')['churn'].mean() * 100
active_churn = active_churn.reindex(active_order)
colors_active = ['#10B981', '#3B82F6', '#F97316', '#EF4444', '#991B1B']
bars5 = ax5.barh(active_churn.index, active_churn.values,
                 color=colors_active, edgecolor='white', linewidth=1.2)
ax5.set_title('Churn Rate by Last Active', fontsize=13, fontweight='bold')
ax5.set_xlabel('Churn Rate (%)', fontsize=11)
ax5.set_xlim(0, 100)
for bar, val in zip(bars5, active_churn.values):
    ax5.text(val + 1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

# ── Plot 6: Churn by Usage Frequency ────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
freq_order = ['Low (1-2)', 'Medium (3-4)', 'High (5-7)']
freq_churn = df.groupby('usage_frequency')['churn'].mean() * 100
freq_churn = freq_churn.reindex(freq_order)
bars6 = ax6.bar(freq_order, freq_churn.values,
                color=['#EF4444', '#F97316', '#10B981'], edgecolor='white', linewidth=1.5, width=0.5)
ax6.set_title('Churn Rate by Usage Frequency', fontsize=13, fontweight='bold')
ax6.set_ylabel('Churn Rate (%)', fontsize=11)
ax6.set_ylim(0, 100)
ax6.set_xticklabels(['Low\n(1-2 days)', 'Medium\n(3-4 days)', 'High\n(5-7 days)'], fontsize=10)
for bar, val in zip(bars6, freq_churn.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.savefig('data/eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[Saved] data/eda_dashboard.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Correlation heatmap (after encoding)
# ══════════════════════════════════════════════════════════════════════════

# Encode for correlation analysis
df_enc = df.copy()

# Ordinal mappings (order matters here)
df_enc['usage_duration']   = df_enc['usage_duration'].map(
    {'0-3 months': 0, '3-6 months': 1, '6-12 months': 2, '12+ months': 3})
df_enc['session_length']   = df_enc['session_length'].map(
    {'<5 min': 0, '5-15 min': 1, '15-30 min': 2, '30+ min': 3})
df_enc['usage_frequency']  = df_enc['usage_frequency'].map(
    {'Low (1-2)': 0, 'Medium (3-4)': 1, 'High (5-7)': 2})
df_enc['subscription_plan'] = df_enc['subscription_plan'].map(
    {'Free': 0, 'Basic': 1, 'Pro': 2, 'Enterprise': 3})
df_enc['last_active']      = df_enc['last_active'].map(
    {'Today': 0, '1-3 days': 1, '4-7 days': 2, '8-14 days': 3, '15+ days': 4})

fig2, ax = plt.subplots(figsize=(10, 8))
corr = df_enc.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle → hide duplicate
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, linewidths=0.5, ax=ax,
            annot_kws={'size': 11, 'weight': 'bold'},
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap\n(with Churn)', fontsize=15, fontweight='bold', pad=16)
plt.xticks(rotation=30, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig('data/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] data/correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════
# KEY INSIGHTS (printed to console)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("KEY INSIGHTS FROM EDA")
print("=" * 60)

print("\n1. CHURN BY SUBSCRIPTION PLAN:")
for plan in plan_order:
    rate = df[df['subscription_plan']==plan]['churn'].mean()*100
    count = len(df[df['subscription_plan']==plan])
    print(f"   {plan:12s}: {rate:.1f}% churn  (n={count})")

print("\n2. CHURN BY EXPERIENCE RATING:")
for r in [1,2,3,4,5]:
    rate = df[df['experience_rating']==r]['churn'].mean()*100
    print(f"   {r} stars: {rate:.1f}% churn")

print("\n3. CHURN BY LAST ACTIVE:")
for status in active_order:
    rate = df[df['last_active']==status]['churn'].mean()*100
    print(f"   {status:12s}: {rate:.1f}% churn")

print("\n4. AVG DAILY ACTIVITY:")
print(f"   Stay  users: {df[df['churn']==0]['daily_activity'].mean():.1f} msgs/day")
print(f"   Churn users: {df[df['churn']==1]['daily_activity'].mean():.1f} msgs/day")

print("\n5. STRONGEST CHURN PREDICTORS (from correlation):")
churn_corr = corr['churn'].drop('churn').abs().sort_values(ascending=False)
for feat, val in churn_corr.items():
    print(f"   {feat:22s}: |r| = {val:.3f}")