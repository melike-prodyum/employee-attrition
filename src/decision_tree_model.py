"""
Employee Attrition Prediction - Decision Tree Model
Çalışan işten ayrılma tahmini için Decision Tree modeli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Ortak utility fonksiyonlarını import et
from data_utils import (
    load_data,
    print_data_info,
    prepare_features,
    get_column_types,
    fill_missing_values,
    apply_one_hot_encoding,
    create_output_directory,
    create_submission_file
)
from evaluation_utils import (
    print_metrics,
    print_classification_report,
    print_confusion_matrix,
    print_feature_importance
)

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("EMPLOYEE ATTRITION PREDICTION - DECISION TREE MODEL")
print("="*70)

# ============================================================================
# 1. VERİ YÜKLEME VE KEŞİF ANALİZİ
# ============================================================================
print("\n[1] Veri Yükleme ve Keşif Analizi")
print("-"*70)

# Veri setlerini yükle
train_df, test_df, submission = load_data()
print_data_info(train_df, test_df)

# ============================================================================
# 2. VERİ ÖN İŞLEME
# ============================================================================
print("\n[2] Veri Ön İşleme")
print("-"*70)

# Features ve target'ı ayır
X_train, X_test, y, train_ids, test_ids = prepare_features(train_df, test_df)

print(f"✓ Feature sayısı: {X_train.shape[1]}")

# Kategorik ve numerik sütunları ayır
categorical_cols, numerical_cols = get_column_types(X_train)

print(f"✓ Kategorik sütunlar ({len(categorical_cols)}): {categorical_cols}")
print(f"✓ Numerik sütunlar ({len(numerical_cols)}): {numerical_cols}")

# Eksik değerleri doldur
X_train, X_test = fill_missing_values(X_train, X_test, categorical_cols, numerical_cols)

# Kategorik değişkenleri One-Hot Encoding ile encode et
X_train, X_test = apply_one_hot_encoding(X_train, X_test, categorical_cols, verbose='detailed')

print(f"\n✓ Veri ön işleme tamamlandı!")
print(f"✓ Train shape: {X_train.shape}")
print(f"✓ Test shape: {X_test.shape}")

# ============================================================================
# 3. DECISION TREE MODELİ OLUŞTURMA
# ============================================================================
print("\n[3] Decision Tree Modeli Oluşturma")
print("-"*70)

# Train-validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train set: {X_train_split.shape[0]} örnekleri")
print(f"✓ Validation set: {X_val_split.shape[0]} örnekleri")

# Decision Tree modeli - model_builders'dan al
from model_builders import build_decision_tree, get_decision_tree_params

dt_params = get_decision_tree_params()
print("\n🌳 Decision Tree parametreleri:")
print(f"  - max_depth: {dt_params['max_depth']} (ağacın maksimum derinliği)")
print(f"  - min_samples_split: {dt_params['min_samples_split']} (dallanma için minimum örnek sayısı)")
print(f"  - min_samples_leaf: {dt_params['min_samples_leaf']} (yaprak düğümdeki minimum örnek sayısı)")
print(f"  - criterion: {dt_params['criterion']} (bölünme kriteri)")
print(f"  - random_state: {dt_params['random_state']}")
print(f"  - class_weight: {dt_params['class_weight']} (dengesiz veri için)")

dt_model = build_decision_tree()

print("\n⏳ Model eğitiliyor...")
dt_model.fit(X_train_split, y_train_split)
print("✓ Model eğitimi tamamlandı!")

# Model bilgileri
print(f"\n📊 Model Özellikleri:")
print(f"  - Ağaç derinliği: {dt_model.get_depth()}")
print(f"  - Yaprak sayısı: {dt_model.get_n_leaves()}")
print(f"  - Toplam düğüm sayısı: {dt_model.tree_.node_count}")

# ============================================================================
# 4. MODEL DEĞERLENDİRME
# ============================================================================
print("\n[4] Model Değerlendirme")
print("-"*70)

# Tahminler
y_train_pred = dt_model.predict(X_train_split)
y_val_pred = dt_model.predict(X_val_split)
y_train_proba = dt_model.predict_proba(X_train_split)[:, 1]
y_val_proba = dt_model.predict_proba(X_val_split)[:, 1]

# Metrikler
print("\n📈 PERFORMANS METRİKLERİ")
print("="*70)

print_metrics(y_train_split, y_train_pred, y_train_proba, 'Train')
print_metrics(y_val_split, y_val_pred, y_val_proba, 'Validation')

# Classification Report
print_classification_report(y_val_split, y_val_pred)

# Confusion Matrix
cm = print_confusion_matrix(y_val_split, y_val_pred)

# Feature Importance
feature_importance = print_feature_importance(dt_model, X_train.columns)

# ============================================================================
# 5. GÖRSELLEŞTİRME
# ============================================================================
print("\n[5] Görselleştirmeler Oluşturuluyor...")
print("-"*70)

# outputs/decision_tree klasörünü oluştur
output_dir = create_output_directory('decision_tree')

# Figure oluştur - Birleşik görsel
fig = plt.figure(figsize=(20, 12))

# 1. Confusion Matrix
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
            xticklabels=['Not Leave', 'Leave'],
            yticklabels=['Not Leave', 'Leave'])
plt.title('Confusion Matrix (Validation)', fontsize=14, fontweight='bold')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin')

# 2. Feature Importance
ax2 = plt.subplot(2, 3, 2)
top_features = feature_importance.head(10)
bars = plt.barh(range(len(top_features)), top_features['Importance'], color='#c0392b', alpha=0.7)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 10 Önemli Özellikler', fontsize=14, fontweight='bold')
# Değerleri çubukların üzerine ekle
for i, (idx, row) in enumerate(top_features.iterrows()):
    plt.text(row['Importance'], i, f' {row["Importance"]:.4f}', 
             va='center', fontsize=9, fontweight='bold')
plt.gca().invert_yaxis()

# 3. ROC Curve
ax3 = plt.subplot(2, 3, 3)
fpr, tpr, _ = roc_curve(y_val_split, y_val_proba)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_val_split, y_val_proba):.4f})', color='#c0392b')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Target Distribution
ax4 = plt.subplot(2, 3, 4)
target_counts = y.value_counts()
plt.bar(['Not Leave (0)', 'Leave (1)'], target_counts.values, color=['#3498db', '#e74c3c'])
plt.title('Target Dağılımı (Train)', fontsize=14, fontweight='bold')
plt.ylabel('Sayı')
for i, v in enumerate(target_counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')

# 5. Accuracy Comparison
ax5 = plt.subplot(2, 3, 5)
metrics_train = [
    accuracy_score(y_train_split, y_train_pred),
    precision_score(y_train_split, y_train_pred),
    recall_score(y_train_split, y_train_pred),
    f1_score(y_train_split, y_train_pred)
]
metrics_val = [
    accuracy_score(y_val_split, y_val_pred),
    precision_score(y_val_split, y_val_pred),
    recall_score(y_val_split, y_val_pred),
    f1_score(y_val_split, y_val_pred)
]
x = np.arange(4)
width = 0.35
bars1 = plt.bar(x - width/2, metrics_train, width, label='Train', color='#e74c3c', alpha=0.7)
bars2 = plt.bar(x + width/2, metrics_val, width, label='Validation', color='#c0392b', alpha=0.9)
# Değerleri çubukların üzerine ekle
for i, (v1, v2) in enumerate(zip(metrics_train, metrics_val)):
    plt.text(i - width/2, v1 + 0.02, f'{v1:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.text(i + width/2, v2 + 0.02, f'{v2:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.xlabel('Metrikler')
plt.ylabel('Skor')
plt.title('Model Performansı Karşılaştırması', fontsize=14, fontweight='bold')
plt.xticks(x, ['Accuracy', 'Precision', 'Recall', 'F1-Score'], rotation=45)
plt.legend()
plt.ylim([0, 1.1])
plt.grid(True, alpha=0.3, axis='y')

# 6. Decision Tree yapısını göster (basitleştirilmiş)
ax6 = plt.subplot(2, 3, 6)
plot_tree(dt_model, 
          max_depth=3,  # Görselleştirme için ilk 3 seviye
          filled=True, 
          feature_names=X_train.columns,
          class_names=['Not Leave', 'Leave'],
          fontsize=7,
          rounded=True)
plt.title('Decision Tree Yapısı (İlk 3 Seviye)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/decision_tree_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Birleşik grafik kaydedildi: outputs/decision_tree/decision_tree_analysis.png")

# ============================================================================
# AYRI AYRI GRAFİKLER
# ============================================================================
print("\n📊 Grafikleri ayrı ayrı kaydediyorum...")

# 1. Confusion Matrix - Ayrı
fig1 = plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
            xticklabels=['Not Leave', 'Leave'],
            yticklabels=['Not Leave', 'Leave'])
plt.title('Confusion Matrix (Validation)', fontsize=14, fontweight='bold')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin')
plt.tight_layout()
plt.savefig(f'{output_dir}/dt_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Confusion Matrix kaydedildi")

# 2. Feature Importance - Ayrı
fig2 = plt.figure(figsize=(10, 8))
top_features = feature_importance.head(10)
bars = plt.barh(range(len(top_features)), top_features['Importance'], color='#c0392b', alpha=0.7)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 10 Önemli Özellikler', fontsize=14, fontweight='bold')
# Değerleri çubukların üzerine ekle
for i, (idx, row) in enumerate(top_features.iterrows()):
    plt.text(row['Importance'], i, f' {row["Importance"]:.4f}', 
             va='center', fontsize=10, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{output_dir}/dt_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Feature Importance kaydedildi")

# 3. ROC Curve - Ayrı
fig3 = plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_val_split, y_val_proba)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_val_split, y_val_proba):.4f})', color='#c0392b')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/dt_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ ROC Curve kaydedildi")

# 4. Target Distribution - Ayrı
fig4 = plt.figure(figsize=(8, 6))
target_counts = y.value_counts()
plt.bar(['Not Leave (0)', 'Leave (1)'], target_counts.values, color=['#3498db', '#e74c3c'])
plt.title('Target Dağılımı (Train)', fontsize=14, fontweight='bold')
plt.ylabel('Sayı')
for i, v in enumerate(target_counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/dt_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Target Distribution kaydedildi")

# 5. Performance Metrics - Ayrı
fig5 = plt.figure(figsize=(10, 6))
metrics_train = [
    accuracy_score(y_train_split, y_train_pred),
    precision_score(y_train_split, y_train_pred),
    recall_score(y_train_split, y_train_pred),
    f1_score(y_train_split, y_train_pred)
]
metrics_val = [
    accuracy_score(y_val_split, y_val_pred),
    precision_score(y_val_split, y_val_pred),
    recall_score(y_val_split, y_val_pred),
    f1_score(y_val_split, y_val_pred)
]
x = np.arange(4)
width = 0.35
bars1 = plt.bar(x - width/2, metrics_train, width, label='Train', color='#e74c3c', alpha=0.7)
bars2 = plt.bar(x + width/2, metrics_val, width, label='Validation', color='#c0392b', alpha=0.9)
# Değerleri çubukların üzerine ekle
for i, (v1, v2) in enumerate(zip(metrics_train, metrics_val)):
    plt.text(i - width/2, v1 + 0.02, f'{v1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.text(i + width/2, v2 + 0.02, f'{v2:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.xlabel('Metrikler')
plt.ylabel('Skor')
plt.title('Model Performansı Karşılaştırması', fontsize=14, fontweight='bold')
plt.xticks(x, ['Accuracy', 'Precision', 'Recall', 'F1-Score'], rotation=45)
plt.legend()
plt.ylim([0, 1.1])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/dt_performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Performance Metrics kaydedildi")

# 6. Tree Structure (simplified) - Ayrı
fig6 = plt.figure(figsize=(20, 12))
plot_tree(dt_model, 
          max_depth=3,
          filled=True, 
          feature_names=X_train.columns,
          class_names=['Not Leave', 'Leave'],
          fontsize=9,
          rounded=True)
plt.title('Decision Tree Yapısı (İlk 3 Seviye)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/dt_tree_structure_simple.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Tree Structure (simplified) kaydedildi")

# Daha detaylı ağaç görselleştirmesi
fig7 = plt.figure(figsize=(25, 15))
plot_tree(dt_model, 
          filled=True, 
          feature_names=X_train.columns,
          class_names=['Not Leave', 'Leave'],
          fontsize=10,
          rounded=True,
          proportion=True)
plt.title('Decision Tree - Tam Yapı', fontsize=16, fontweight='bold', pad=20)
plt.savefig(f'{output_dir}/decision_tree_full.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Tam ağaç görselleştirmesi kaydedildi")

print("\n✓ Tüm grafikler hem birleşik hem de ayrı ayrı kaydedildi!")

# ============================================================================
# 6. TEST VERİSİ İÇİN TAHMİNLER
# ============================================================================
print("\n[6] Test Verisi Tahminleri")
print("-"*70)

# Tüm train verisi ile son modeli eğit
print("⏳ Final model tüm train verisi ile eğitiliyor...")
final_model = build_decision_tree()
final_model.fit(X_train, y)
print("✓ Final model eğitimi tamamlandı!")

# Submission dosyasını hazırla
test_predictions, submission_df = create_submission_file(
    final_model, X_test, submission, 'submission_decision_tree.csv'
)
print(f"✓ Submission dosyası oluşturuldu: submissions/submission_decision_tree.csv")
print(f"✓ Tahmin edilen test örnekleri: {len(test_predictions)}")
print(f"\nTahmin İstatistikleri:")
print(f"  • Ortalama: {test_predictions.mean():.4f}")
print(f"  • Std:      {test_predictions.std():.4f}")
print(f"  • Min:      {test_predictions.min():.4f}")
print(f"  • Max:      {test_predictions.max():.4f}")

# ============================================================================
# ÖZET
# ============================================================================
print("\n" + "="*70)
print("MODEL ÖZETI")
print("="*70)
print(f"\n✓ Model Tipi: Decision Tree Classifier")
print(f"✓ Ağaç Derinliği: {final_model.get_depth()}")
print(f"✓ Yaprak Sayısı: {final_model.get_n_leaves()}")
print(f"✓ Validation Accuracy: {accuracy_score(y_val_split, y_val_pred):.4f}")
print(f"✓ Validation ROC-AUC: {roc_auc_score(y_val_split, y_val_proba):.4f}")
print(f"\n📁 Oluşturulan Dosyalar:")
print(f"  Birleşik Görsel:")
print(f"    • outputs/decision_tree/decision_tree_analysis.png - Genel analiz grafikleri")
print(f"  Ayrı Görseller:")
print(f"    • outputs/decision_tree/dt_confusion_matrix.png")
print(f"    • outputs/decision_tree/dt_feature_importance.png")
print(f"    • outputs/decision_tree/dt_roc_curve.png")
print(f"    • outputs/decision_tree/dt_target_distribution.png")
print(f"    • outputs/decision_tree/dt_performance_metrics.png")
print(f"    • outputs/decision_tree/dt_tree_structure_simple.png")
print(f"    • outputs/decision_tree/decision_tree_full.png")
print(f"  Submission:")
print(f"    • submissions/submission_decision_tree.csv")

print("\n" + "="*70)
print("✅ İŞLEM TAMAMLANDI!")
print("="*70)
print("\n💡 Not: Decision Tree basit ve yorumlanabilir bir modeldir.")
print("   Random Forest ile karşılaştırma yapmak için birden fazla")
print("   ağacın ensemble'ını kullanmanız gerekecek.")
print("="*70)
