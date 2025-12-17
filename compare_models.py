"""
Employee Attrition Prediction - Random Forest vs Decision Tree Comparison
Çalışan işten ayrılma tahmini - Random Forest ve Decision Tree karşılaştırması
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("RANDOM FOREST VS DECISION TREE - KARŞILAŞTIRMA")
print("="*70)

# ============================================================================
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# ============================================================================
print("\n[1] Veri Yükleme ve Ön İşleme")
print("-"*70)

train_df = pd.read_csv('aug_train.csv')
test_df = pd.read_csv('aug_test.csv')

y = train_df['target']
X_train = train_df.drop(['enrollee_id', 'target'], axis=1)
X_test = test_df.drop(['enrollee_id'], axis=1)

# Kategorik ve numerik sütunları ayır
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Eksik değerleri doldur
for col in numerical_cols:
    if X_train[col].isnull().sum() > 0:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

for col in categorical_cols:
    if X_train[col].isnull().sum() > 0:
        mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else 'Unknown'
        X_train[col].fillna(mode_val, inplace=True)
        X_test[col].fillna(mode_val, inplace=True)

# Kategorik değişkenleri encode et
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Train-validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Veri hazır - Train: {X_train_split.shape}, Validation: {X_val_split.shape}")

# ============================================================================
# 2. DECISION TREE MODELİ
# ============================================================================
print("\n[2] Decision Tree Modeli")
print("-"*70)

dt_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=100,
    min_samples_leaf=50,
    criterion='gini',
    random_state=42,
    class_weight='balanced'
)

print("⏳ Decision Tree eğitiliyor...")
dt_model.fit(X_train_split, y_train_split)
print("✓ Eğitim tamamlandı!")

# Tahminler
dt_train_pred = dt_model.predict(X_train_split)
dt_val_pred = dt_model.predict(X_val_split)
dt_train_proba = dt_model.predict_proba(X_train_split)[:, 1]
dt_val_proba = dt_model.predict_proba(X_val_split)[:, 1]

# Metrikler
dt_metrics = {
    'Model': 'Decision Tree',
    'Train Accuracy': accuracy_score(y_train_split, dt_train_pred),
    'Val Accuracy': accuracy_score(y_val_split, dt_val_pred),
    'Train Precision': precision_score(y_train_split, dt_train_pred),
    'Val Precision': precision_score(y_val_split, dt_val_pred),
    'Train Recall': recall_score(y_train_split, dt_train_pred),
    'Val Recall': recall_score(y_val_split, dt_val_pred),
    'Train F1': f1_score(y_train_split, dt_train_pred),
    'Val F1': f1_score(y_val_split, dt_val_pred),
    'Train ROC-AUC': roc_auc_score(y_train_split, dt_train_proba),
    'Val ROC-AUC': roc_auc_score(y_val_split, dt_val_proba),
    'Tree Depth': dt_model.get_depth(),
    'Leaves': dt_model.get_n_leaves()
}

print(f"\n📊 Decision Tree Sonuçları:")
print(f"  Validation Accuracy:  {dt_metrics['Val Accuracy']:.4f}")
print(f"  Validation ROC-AUC:   {dt_metrics['Val ROC-AUC']:.4f}")
print(f"  Ağaç Derinliği:       {dt_metrics['Tree Depth']}")
print(f"  Yaprak Sayısı:        {dt_metrics['Leaves']}")

# ============================================================================
# 3. RANDOM FOREST MODELİ
# ============================================================================
print("\n[3] Random Forest Modeli")
print("-"*70)

rf_model = RandomForestClassifier(
    n_estimators=100,        # 100 ağaç
    max_depth=6,             # Her ağaç için maksimum derinlik
    min_samples_split=100,
    min_samples_leaf=50,
    criterion='gini',
    random_state=42,
    class_weight='balanced',
    n_jobs=-1                # Paralel işleme
)

print("⏳ Random Forest eğitiliyor (100 ağaç)...")
rf_model.fit(X_train_split, y_train_split)
print("✓ Eğitim tamamlandı!")

# Tahminler
rf_train_pred = rf_model.predict(X_train_split)
rf_val_pred = rf_model.predict(X_val_split)
rf_train_proba = rf_model.predict_proba(X_train_split)[:, 1]
rf_val_proba = rf_model.predict_proba(X_val_split)[:, 1]

# Metrikler
rf_metrics = {
    'Model': 'Random Forest',
    'Train Accuracy': accuracy_score(y_train_split, rf_train_pred),
    'Val Accuracy': accuracy_score(y_val_split, rf_val_pred),
    'Train Precision': precision_score(y_train_split, rf_train_pred),
    'Val Precision': precision_score(y_val_split, rf_val_pred),
    'Train Recall': recall_score(y_train_split, rf_train_pred),
    'Val Recall': recall_score(y_val_split, rf_val_pred),
    'Train F1': f1_score(y_train_split, rf_train_pred),
    'Val F1': f1_score(y_val_split, rf_val_pred),
    'Train ROC-AUC': roc_auc_score(y_train_split, rf_train_proba),
    'Val ROC-AUC': roc_auc_score(y_val_split, rf_val_proba),
    'N Trees': rf_model.n_estimators
}

print(f"\n📊 Random Forest Sonuçları:")
print(f"  Validation Accuracy:  {rf_metrics['Val Accuracy']:.4f}")
print(f"  Validation ROC-AUC:   {rf_metrics['Val ROC-AUC']:.4f}")
print(f"  Ağaç Sayısı:          {rf_metrics['N Trees']}")

# ============================================================================
# 4. KARŞILAŞTIRMA
# ============================================================================
print("\n[4] Model Karşılaştırması")
print("="*70)

comparison_df = pd.DataFrame([
    {
        'Model': 'Decision Tree',
        'Accuracy': dt_metrics['Val Accuracy'],
        'Precision': dt_metrics['Val Precision'],
        'Recall': dt_metrics['Val Recall'],
        'F1-Score': dt_metrics['Val F1'],
        'ROC-AUC': dt_metrics['Val ROC-AUC']
    },
    {
        'Model': 'Random Forest',
        'Accuracy': rf_metrics['Val Accuracy'],
        'Precision': rf_metrics['Val Precision'],
        'Recall': rf_metrics['Val Recall'],
        'F1-Score': rf_metrics['Val F1'],
        'ROC-AUC': rf_metrics['Val ROC-AUC']
    }
])

print("\n📊 Validation Metrikleri Karşılaştırması:")
print(comparison_df.to_string(index=False))

# Fark hesapla
print("\n🔍 Random Forest İyileşmeleri:")
metric_map = {
    'Accuracy': 'Accuracy',
    'Precision': 'Precision',
    'Recall': 'Recall',
    'F1': 'F1-Score',
    'ROC-AUC': 'ROC-AUC'
}
for short_name, display_name in metric_map.items():
    dt_val = dt_metrics[f'Val {short_name}']
    rf_val = rf_metrics[f'Val {short_name}']
    diff = rf_val - dt_val
    pct = (diff / dt_val) * 100 if dt_val != 0 else 0
    symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
    print(f"  {symbol} {display_name:12s}: {diff:+.4f} ({pct:+.2f}%)")

# ============================================================================
# 5. GÖRSELLEŞTİRME
# ============================================================================
print("\n[5] Karşılaştırma Grafikleri")
print("-"*70)

fig = plt.figure(figsize=(18, 12))

# 1. Metrics Comparison
ax1 = plt.subplot(2, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
dt_vals = [dt_metrics['Val Accuracy'], dt_metrics['Val Precision'], 
           dt_metrics['Val Recall'], dt_metrics['Val F1'], dt_metrics['Val ROC-AUC']]
rf_vals = [rf_metrics['Val Accuracy'], rf_metrics['Val Precision'], 
           rf_metrics['Val Recall'], rf_metrics['Val F1'], rf_metrics['Val ROC-AUC']]

x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, dt_vals, width, label='Decision Tree', color='#e74c3c', alpha=0.8)
plt.bar(x + width/2, rf_vals, width, label='Random Forest', color='#2ecc71', alpha=0.8)
plt.xlabel('Metrikler', fontweight='bold')
plt.ylabel('Skor', fontweight='bold')
plt.title('Model Performansı Karşılaştırması (Validation)', fontsize=12, fontweight='bold')
plt.xticks(x, metrics, rotation=45, ha='right')
plt.legend()
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')

# 2. ROC Curves
ax2 = plt.subplot(2, 3, 2)
dt_fpr, dt_tpr, _ = roc_curve(y_val_split, dt_val_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_val_split, rf_val_proba)

plt.plot(dt_fpr, dt_tpr, linewidth=2, label=f'Decision Tree (AUC={dt_metrics["Val ROC-AUC"]:.4f})', color='#e74c3c')
plt.plot(rf_fpr, rf_tpr, linewidth=2, label=f'Random Forest (AUC={rf_metrics["Val ROC-AUC"]:.4f})', color='#2ecc71')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('ROC Curve Karşılaştırması', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Confusion Matrices
ax3 = plt.subplot(2, 3, 3)
dt_cm = confusion_matrix(y_val_split, dt_val_pred)
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Reds', cbar=False, alpha=0.8)
plt.title('Decision Tree\nConfusion Matrix', fontsize=12, fontweight='bold')
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')

ax4 = plt.subplot(2, 3, 4)
rf_cm = confusion_matrix(y_val_split, rf_val_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', cbar=False, alpha=0.8)
plt.title('Random Forest\nConfusion Matrix', fontsize=12, fontweight='bold')
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')

# 4. Feature Importance Comparison
ax5 = plt.subplot(2, 3, 5)
dt_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'DT_Importance': dt_model.feature_importances_,
    'RF_Importance': rf_model.feature_importances_
}).sort_values('RF_Importance', ascending=False).head(8)

x = np.arange(len(dt_importance))
width = 0.35
plt.barh(x - width/2, dt_importance['DT_Importance'], width, label='Decision Tree', color='#e74c3c', alpha=0.8)
plt.barh(x + width/2, dt_importance['RF_Importance'], width, label='Random Forest', color='#2ecc71', alpha=0.8)
plt.yticks(x, dt_importance['Feature'])
plt.xlabel('Importance', fontweight='bold')
plt.title('Top 8 Feature Importance', fontsize=12, fontweight='bold')
plt.legend()
plt.gca().invert_yaxis()

# 5. Overfitting Comparison
ax6 = plt.subplot(2, 3, 6)
models = ['Decision Tree', 'Random Forest']
train_scores = [dt_metrics['Train Accuracy'], rf_metrics['Train Accuracy']]
val_scores = [dt_metrics['Val Accuracy'], rf_metrics['Val Accuracy']]
overfit = [train_scores[0] - val_scores[0], train_scores[1] - val_scores[1]]

x = np.arange(len(models))
width = 0.25
plt.bar(x - width, train_scores, width, label='Train', color='#3498db')
plt.bar(x, val_scores, width, label='Validation', color='#f39c12')
plt.bar(x + width, overfit, width, label='Overfitting Gap', color='#95a5a6')
plt.xlabel('Model', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Overfitting Analizi', fontsize=12, fontweight='bold')
plt.xticks(x, models)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Karşılaştırma grafiği kaydedildi: model_comparison.png")

# ============================================================================
# 6. RANDOM FOREST İLE TEST TAHMİNLERİ
# ============================================================================
print("\n[6] Random Forest ile Test Tahminleri")
print("-"*70)

# Final Random Forest modeli
final_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_split=100,
    min_samples_leaf=50,
    criterion='gini',
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

print("⏳ Final Random Forest modeli eğitiliyor...")
final_rf.fit(X_train, y)
print("✓ Eğitim tamamlandı!")

# Test tahminleri
rf_test_predictions = final_rf.predict_proba(X_test)[:, 1]

# Submission dosyası
submission = pd.read_csv('sample_submission.csv')
submission['target'] = rf_test_predictions
submission.to_csv('submission_random_forest.csv', index=False)
print(f"✓ Random Forest submission dosyası: submission_random_forest.csv")

# ============================================================================
# ÖZET
# ============================================================================
print("\n" + "="*70)
print("KARŞILAŞTIRMA ÖZETİ")
print("="*70)

print("\n🌳 DECISION TREE:")
print(f"  • Tek ağaç kullanır")
print(f"  • Yorumlanabilir ve basit")
print(f"  • Hızlı eğitim")
print(f"  • Overfitting riski yüksek")
print(f"  • Ağaç derinliği: {dt_metrics['Tree Depth']}")
print(f"  • Yaprak sayısı: {dt_metrics['Leaves']}")
print(f"  • Validation Accuracy: {dt_metrics['Val Accuracy']:.4f}")
print(f"  • Validation ROC-AUC: {dt_metrics['Val ROC-AUC']:.4f}")

print("\n🌲 RANDOM FOREST:")
print(f"  • {rf_metrics['N Trees']} ağaç kullanır (ensemble)")
print(f"  • Daha robust ve güçlü")
print(f"  • Daha yavaş eğitim")
print(f"  • Overfitting riski düşük")
print(f"  • Validation Accuracy: {rf_metrics['Val Accuracy']:.4f}")
print(f"  • Validation ROC-AUC: {rf_metrics['Val ROC-AUC']:.4f}")

print("\n💡 TEMEL FARKLAR:")
print("  1. Decision Tree tek ağaç, Random Forest birçok ağaçtan oluşur")
print("  2. Random Forest her ağacı farklı veri örnekleri ile eğitir")
print("  3. Random Forest her dallanmada rastgele feature seçer")
print("  4. Random Forest tahminleri tüm ağaçların ortalamasıdır")
print("  5. Random Forest genellikle daha yüksek doğruluk sağlar")
print("  6. Decision Tree daha yorumlanabilir")

print("\n📁 Oluşturulan Dosyalar:")
print("  • model_comparison.png - Model karşılaştırma grafikleri")
print("  • submission_random_forest.csv - Random Forest tahminleri")

print("\n" + "="*70)
print("✅ KARŞILAŞTIRMA TAMAMLANDI!")
print("="*70)
