"""
Employee Attrition Prediction - Random Forest vs Decision Tree Comparison
Çalışan işten ayrılma tahmini - Random Forest ve Decision Tree karşılaştırması
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

# Model builder fonksiyonlarını import et
from model_builders import (
    build_decision_tree,
    build_random_forest,
    get_decision_tree_params,
    get_random_forest_params
)

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

train_df = pd.read_csv('../data/aug_train.csv')
test_df = pd.read_csv('../data/aug_test.csv')

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
# Decision Tree için One-Hot Encoding (decision_tree_model.py ile aynı)
# Random Forest için Label Encoding (random_forest_model.py ile aynı)
# Karşılaştırma için One-Hot Encoding kullanacağız (daha iyi sonuçlar için)
print("\n🔧 Kategorik değişkenleri One-Hot Encoding ile encode etme:")

if categorical_cols:
    # One-Hot Encoding uygula
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)
    
    # Train ve test'te aynı sütunların olmasını sağla
    missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    for col in missing_cols:
        X_test_encoded[col] = 0
    
    extra_cols = set(X_test_encoded.columns) - set(X_train_encoded.columns)
    X_test_encoded = X_test_encoded.drop(columns=extra_cols)
    
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    X_train = X_train_encoded
    X_test = X_test_encoded
    
    print(f"  ✓ One-Hot Encoding tamamlandı - {X_train.shape[1]} feature")

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

# Model ve parametreleri decision_tree_model.py'den al
dt_params = get_decision_tree_params()
print("🌳 Decision Tree parametreleri:")
print(f"  - max_depth: {dt_params['max_depth']} (ağacın maksimum derinliği)")
print(f"  - min_samples_split: {dt_params['min_samples_split']} (dallanma için minimum örnek sayısı)")
print(f"  - min_samples_leaf: {dt_params['min_samples_leaf']} (yaprak düğümdeki minimum örnek sayısı)")
print(f"  - criterion: {dt_params['criterion']} (bölünme kriteri)")
print(f"  - random_state: {dt_params['random_state']}")
print(f"  - class_weight: {dt_params['class_weight']} (dengesiz veri için)")

dt_model = build_decision_tree()

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

# Model ve parametreleri random_forest_model.py'den al
rf_params = get_random_forest_params()
print("🌲 Random Forest parametreleri:")
print(f"  - n_estimators: {rf_params['n_estimators']} ({rf_params['n_estimators']} farklı decision tree)")
print(f"  - max_depth: {rf_params['max_depth']} (her ağacın maksimum derinliği - basit)")
print(f"  - min_samples_split: {rf_params['min_samples_split']} (dallanma için minimum örnek)")
print(f"  - min_samples_leaf: {rf_params['min_samples_leaf']} (yaprak düğümdeki minimum örnek)")
print(f"  - criterion: {rf_params['criterion']} (bölünme kriteri)")
print(f"  - random_state: {rf_params['random_state']}")
print(f"  - class_weight: {rf_params['class_weight']} (dengesiz veri için)")
print(f"  - n_jobs: {rf_params['n_jobs']} (paralel işleme)")
print(f"  - max_features: {rf_params['max_features']} (her dallanmada rastgele feature seç)")

rf_model = build_random_forest()

print("\n⏳ Random Forest eğitiliyor (100 ağaç)...")
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

# outputs klasörünü oluştur
import os
os.makedirs('../outputs', exist_ok=True)

# Birleşik görsel
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
plt.savefig('../outputs/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Birleşik karşılaştırma grafiği kaydedildi: outputs/model_comparison.png")

# ============================================================================
# AYRI AYRI GRAFİKLER
# ============================================================================
print("\n📊 Grafikleri ayrı ayrı kaydediyorum...")

# 1. Metrics Comparison - Ayrı
fig1 = plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
dt_vals = [dt_metrics['Val Accuracy'], dt_metrics['Val Precision'], 
           dt_metrics['Val Recall'], dt_metrics['Val F1'], dt_metrics['Val ROC-AUC']]
rf_vals = [rf_metrics['Val Accuracy'], rf_metrics['Val Precision'], 
           rf_metrics['Val Recall'], rf_metrics['Val F1'], rf_metrics['Val ROC-AUC']]
x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, dt_vals, width, label='Decision Tree', color='#c0392b', alpha=0.8)
plt.bar(x + width/2, rf_vals, width, label='Random Forest', color='#27ae60', alpha=0.8)
plt.xlabel('Metrikler', fontweight='bold')
plt.ylabel('Skor', fontweight='bold')
plt.title('Model Performansı Karşılaştırması (Validation)', fontsize=14, fontweight='bold')
plt.xticks(x, metrics, rotation=45, ha='right')
plt.legend()
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../outputs/compare_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Metrics Comparison kaydedildi")

# 2. ROC Curves - Ayrı
fig2 = plt.figure(figsize=(8, 6))
dt_fpr, dt_tpr, _ = roc_curve(y_val_split, dt_val_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_val_split, rf_val_proba)
plt.plot(dt_fpr, dt_tpr, linewidth=2, label=f'Decision Tree (AUC={dt_metrics["Val ROC-AUC"]:.4f})', color='#c0392b')
plt.plot(rf_fpr, rf_tpr, linewidth=2, label=f'Random Forest (AUC={rf_metrics["Val ROC-AUC"]:.4f})', color='#27ae60')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('ROC Curve Karşılaştırması', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/compare_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ ROC Curves kaydedildi")

# 3. Decision Tree Confusion Matrix - Ayrı
fig3 = plt.figure(figsize=(8, 6))
dt_cm = confusion_matrix(y_val_split, dt_val_pred)
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Reds', cbar=False, alpha=0.8,
            xticklabels=['Not Leave', 'Leave'],
            yticklabels=['Not Leave', 'Leave'])
plt.title('Decision Tree - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin')
plt.tight_layout()
plt.savefig('../outputs/compare_dt_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Decision Tree Confusion Matrix kaydedildi")

# 4. Random Forest Confusion Matrix - Ayrı
fig4 = plt.figure(figsize=(8, 6))
rf_cm = confusion_matrix(y_val_split, rf_val_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', cbar=False, alpha=0.8,
            xticklabels=['Not Leave', 'Leave'],
            yticklabels=['Not Leave', 'Leave'])
plt.title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin')
plt.tight_layout()
plt.savefig('../outputs/compare_rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Random Forest Confusion Matrix kaydedildi")

# 5. Feature Importance - Ayrı
fig5 = plt.figure(figsize=(10, 8))
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
plt.title('Top 8 Feature Importance Karşılaştırması', fontsize=14, fontweight='bold')
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../outputs/compare_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Feature Importance kaydedildi")

# 6. Overfitting Analysis - Ayrı
fig6 = plt.figure(figsize=(10, 6))
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
plt.title('Overfitting Analizi', fontsize=14, fontweight='bold')
plt.xticks(x, models)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../outputs/compare_overfitting.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Overfitting Analysis kaydedildi")

print("\n✓ Tüm grafikler hem birleşik hem de ayrı ayrı kaydedildi!")


# ============================================================================
# 6. RANDOM FOREST İLE TEST TAHMİNLERİ
# ============================================================================
print("\n[6] Random Forest ile Test Tahminleri")
print("-"*70)

# Final Random Forest modeli - random_forest_model.py'den al
final_rf = build_random_forest()

print("⏳ Final Random Forest modeli eğitiliyor...")
final_rf.fit(X_train, y)
print("✓ Eğitim tamamlandı!")

# Test tahminleri
rf_test_predictions = final_rf.predict_proba(X_test)[:, 1]

# Submission dosyası
submission = pd.read_csv('../data/sample_submission.csv')
submission['target'] = rf_test_predictions
os.makedirs('../submissions', exist_ok=True)
submission.to_csv('../submissions/submission_random_forest.csv', index=False)
print(f"✓ Random Forest submission dosyası: submissions/submission_random_forest.csv")

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
print("  Birleşik Görsel:")
print("    • outputs/model_comparison.png - Model karşılaştırma grafikleri")
print("  Ayrı Görseller:")
print("    • outputs/compare_metrics.png")
print("    • outputs/compare_roc_curves.png")
print("    • outputs/compare_dt_confusion_matrix.png")
print("    • outputs/compare_rf_confusion_matrix.png")
print("    • outputs/compare_feature_importance.png")
print("    • outputs/compare_overfitting.png")
print("  Submission:")
print("    • submissions/submission_random_forest.csv")

print("\n" + "="*70)
print("✅ KARŞILAŞTIRMA TAMAMLANDI!")
print("="*70)
