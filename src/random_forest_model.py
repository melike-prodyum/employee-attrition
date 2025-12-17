"""
Employee Attrition Prediction - Random Forest Model
Çalışan işten ayrılma tahmini için Random Forest modeli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("EMPLOYEE ATTRITION PREDICTION - RANDOM FOREST MODEL")
print("="*70)

# ============================================================================
# 1. VERİ YÜKLEME VE KEŞİF ANALİZİ
# ============================================================================
print("\n[1] Veri Yükleme ve Keşif Analizi")
print("-"*70)

# Veri setlerini yükle
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_df = pd.read_csv(os.path.join(project_root, 'data', 'aug_train.csv'))
test_df = pd.read_csv(os.path.join(project_root, 'data', 'aug_test.csv'))
submission = pd.read_csv(os.path.join(project_root, 'data', 'sample_submission.csv'))

print(f"✓ Train veri seti boyutu: {train_df.shape}")
print(f"✓ Test veri seti boyutu: {test_df.shape}")
print(f"\nSütunlar: {list(train_df.columns)}")

# Target dağılımı
print(f"\n📊 Target Dağılımı:")
print(train_df['target'].value_counts())
print(f"Target oranı: {train_df['target'].value_counts(normalize=True)}")

# Eksik değerler
print(f"\n📋 Eksik Değerler:")
missing = train_df.isnull().sum()
missing_pct = (missing / len(train_df)) * 100
missing_df = pd.DataFrame({
    'Eksik Sayısı': missing,
    'Yüzde': missing_pct
}).sort_values('Eksik Sayısı', ascending=False)
print(missing_df[missing_df['Eksik Sayısı'] > 0])

# ============================================================================
# 2. VERİ ÖN İŞLEME
# ============================================================================
print("\n[2] Veri Ön İşleme")
print("-"*70)

# enrollee_id'yi ayır
train_ids = train_df['enrollee_id']
test_ids = test_df['enrollee_id']

# Target değişkeni ayır
y = train_df['target']
X_train = train_df.drop(['enrollee_id', 'target'], axis=1)
X_test = test_df.drop(['enrollee_id'], axis=1)

print(f"✓ Feature sayısı: {X_train.shape[1]}")

# Kategorik ve numerik sütunları ayır
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"✓ Kategorik sütunlar ({len(categorical_cols)}): {categorical_cols}")
print(f"✓ Numerik sütunlar ({len(numerical_cols)}): {numerical_cols}")

# Eksik değerleri doldur
print("\n🔧 Eksik değerleri doldurma:")

# Numerik sütunlar için median
for col in numerical_cols:
    if X_train[col].isnull().sum() > 0:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
        print(f"  - {col}: median ile dolduruldu")

# Kategorik sütunlar için mode
for col in categorical_cols:
    if X_train[col].isnull().sum() > 0:
        mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else 'Unknown'
        X_train[col].fillna(mode_val, inplace=True)
        X_test[col].fillna(mode_val, inplace=True)
        print(f"  - {col}: mode ile dolduruldu")

# Kategorik değişkenleri encode et
print("\n🔧 Kategorik değişkenleri encode etme:")
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # Train ve test'i birleştirerek tüm kategorileri öğren
    combined = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(combined)
    
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le
    print(f"  - {col}: {len(le.classes_)} kategori")

print(f"\n✓ Veri ön işleme tamamlandı!")
print(f"✓ Train shape: {X_train.shape}")
print(f"✓ Test shape: {X_test.shape}")

# ============================================================================
# 3. RANDOM FOREST MODELİ OLUŞTURMA
# ============================================================================
print("\n[3] Random Forest Modeli Oluşturma")
print("-"*70)

# Train-validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train set: {X_train_split.shape[0]} örnekleri")
print(f"✓ Validation set: {X_val_split.shape[0]} örnekleri")

# Random Forest modeli - Her ağaç basit ama birçok ağaç
print("\n🌲 Random Forest parametreleri:")
print("  - n_estimators: 100 (100 farklı decision tree)")
print("  - max_depth: 4 (her ağacın maksimum derinliği - basit)")
print("  - min_samples_split: 200 (dallanma için minimum örnek)")
print("  - min_samples_leaf: 100 (yaprak düğümdeki minimum örnek)")
print("  - criterion: gini (bölünme kriteri)")
print("  - random_state: 42")
print("  - n_jobs: -1 (paralel işleme)")

rf_model = RandomForestClassifier(
    n_estimators=100,               # 100 ağaç oluştur
    max_depth=4,                    # Her ağaç basit olsun (4 seviye)
    min_samples_split=200,          # Dallanma için gereken minimum örnek
    min_samples_leaf=100,           # Her yaprakta en az bu kadar örnek
    criterion='gini',               # Gini impurity kullan
    random_state=42,
    class_weight='balanced',        # Dengesiz veri için sınıf ağırlıkları
    n_jobs=-1,                      # Tüm CPU core'ları kullan
    max_features='sqrt'             # Her dallanmada rastgele feature seç
)

print("\n⏳ Random Forest eğitiliyor (100 ağaç)...")
rf_model.fit(X_train_split, y_train_split)
print("✓ Model eğitimi tamamlandı!")

# Model bilgileri
print(f"\n📊 Model Özellikleri:")
print(f"  - Ağaç sayısı: {rf_model.n_estimators}")
print(f"  - Her ağaç için max derinlik: {rf_model.max_depth}")
print(f"  - Toplam estimator: {len(rf_model.estimators_)}")

# İlk birkaç ağacın derinliğini göster
tree_depths = [tree.get_depth() for tree in rf_model.estimators_[:5]]
print(f"  - İlk 5 ağacın derinlikleri: {tree_depths}")

# ============================================================================
# 4. MODEL DEĞERLENDİRME
# ============================================================================
print("\n[4] Model Değerlendirme")
print("-"*70)

# Tahminler
y_train_pred = rf_model.predict(X_train_split)
y_val_pred = rf_model.predict(X_val_split)
y_train_proba = rf_model.predict_proba(X_train_split)[:, 1]
y_val_proba = rf_model.predict_proba(X_val_split)[:, 1]

# Metrikler
print("\n📈 PERFORMANS METRİKLERİ")
print("="*70)

print("\n🔹 Train Seti:")
print(f"  • Accuracy:  {accuracy_score(y_train_split, y_train_pred):.4f}")
print(f"  • Precision: {precision_score(y_train_split, y_train_pred):.4f}")
print(f"  • Recall:    {recall_score(y_train_split, y_train_pred):.4f}")
print(f"  • F1-Score:  {f1_score(y_train_split, y_train_pred):.4f}")
print(f"  • ROC-AUC:   {roc_auc_score(y_train_split, y_train_proba):.4f}")

print("\n🔹 Validation Seti:")
print(f"  • Accuracy:  {accuracy_score(y_val_split, y_val_pred):.4f}")
print(f"  • Precision: {precision_score(y_val_split, y_val_pred):.4f}")
print(f"  • Recall:    {recall_score(y_val_split, y_val_pred):.4f}")
print(f"  • F1-Score:  {f1_score(y_val_split, y_val_pred):.4f}")
print(f"  • ROC-AUC:   {roc_auc_score(y_val_split, y_val_proba):.4f}")

# Classification Report
print("\n📋 Detaylı Sınıflandırma Raporu (Validation):")
print("-"*70)
print(classification_report(y_val_split, y_val_pred, 
                          target_names=['Not Leave (0)', 'Leave (1)']))

# Confusion Matrix
cm = confusion_matrix(y_val_split, y_val_pred)
print("\n🔢 Confusion Matrix (Validation):")
print(cm)
print(f"  True Negatives:  {cm[0, 0]}")
print(f"  False Positives: {cm[0, 1]}")
print(f"  False Negatives: {cm[1, 0]}")
print(f"  True Positives:  {cm[1, 1]}")

# Feature Importance
print("\n⭐ En Önemli Özellikler (Top 10):")
print("-"*70)
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']:30s} : {row['Importance']:.4f}")

# ============================================================================
# 5. GÖRSELLEŞTİRME
# ============================================================================
print("\n[5] Görselleştirmeler Oluşturuluyor...")
print("-"*70)

# outputs/random_forest klasörünü oluştur
import os
os.makedirs('../outputs/random_forest', exist_ok=True)

# Figure oluştur - Birleşik görsel
fig = plt.figure(figsize=(20, 14))

# 1. Confusion Matrix
ax1 = plt.subplot(3, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Not Leave', 'Leave'],
            yticklabels=['Not Leave', 'Leave'])
plt.title('Confusion Matrix (Validation)', fontsize=14, fontweight='bold')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin')

# 2. Feature Importance
ax2 = plt.subplot(3, 3, 2)
top_features = feature_importance.head(10)
bars = plt.barh(range(len(top_features)), top_features['Importance'], color='#27ae60', alpha=0.7)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 10 Önemli Özellikler', fontsize=14, fontweight='bold')
# Değerleri çubukların üzerine ekle
for i, (idx, row) in enumerate(top_features.iterrows()):
    plt.text(row['Importance'], i, f' {row["Importance"]:.4f}', 
             va='center', fontsize=9, fontweight='bold')
plt.gca().invert_yaxis()

# 3. ROC Curve
ax3 = plt.subplot(3, 3, 3)
fpr, tpr, _ = roc_curve(y_val_split, y_val_proba)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_val_split, y_val_proba):.4f})', color='#27ae60')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Target Distribution
ax4 = plt.subplot(3, 3, 4)
target_counts = y.value_counts()
plt.bar(['Not Leave (0)', 'Leave (1)'], target_counts.values, color=['#3498db', '#e74c3c'])
plt.title('Target Dağılımı (Train)', fontsize=14, fontweight='bold')
plt.ylabel('Sayı')
for i, v in enumerate(target_counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')

# 5. Accuracy Comparison
ax5 = plt.subplot(3, 3, 5)
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
bars1 = plt.bar(x - width/2, metrics_train, width, label='Train', color='#2ecc71', alpha=0.7)
bars2 = plt.bar(x + width/2, metrics_val, width, label='Validation', color='#27ae60', alpha=0.9)
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

# 6-9. İlk 4 ağacın görselleştirmesi
print("✓ İlk 4 ağacı görselleştiriyorum...")
for i in range(4):
    ax = plt.subplot(3, 3, 6 + i)
    plot_tree(rf_model.estimators_[i], 
              max_depth=2,  # Sadece ilk 2 seviye göster
              filled=True, 
              feature_names=X_train.columns,
              class_names=['Not Leave', 'Leave'],
              fontsize=7,
              rounded=True)
    plt.title(f'Ağaç #{i+1} (İlk 2 Seviye)', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/random_forest/random_forest_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Birleşik grafik kaydedildi: outputs/random_forest/random_forest_analysis.png")

# ============================================================================
# AYRI AYRI GRAFİKLER
# ============================================================================
print("\n📊 Grafikleri ayrı ayrı kaydediyorum...")

# 1. Confusion Matrix - Ayrı
fig1 = plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Not Leave', 'Leave'],
            yticklabels=['Not Leave', 'Leave'])
plt.title('Confusion Matrix (Validation)', fontsize=14, fontweight='bold')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin')
plt.tight_layout()
plt.savefig('../outputs/random_forest/rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Confusion Matrix kaydedildi")

# 2. Feature Importance - Ayrı
fig2 = plt.figure(figsize=(10, 8))
top_features = feature_importance.head(10)
bars = plt.barh(range(len(top_features)), top_features['Importance'], color='#27ae60', alpha=0.7)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 10 Önemli Özellikler', fontsize=14, fontweight='bold')
# Değerleri çubukların üzerine ekle
for i, (idx, row) in enumerate(top_features.iterrows()):
    plt.text(row['Importance'], i, f' {row["Importance"]:.4f}', 
             va='center', fontsize=10, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../outputs/random_forest/rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Feature Importance kaydedildi")

# 3. ROC Curve - Ayrı
fig3 = plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_val_split, y_val_proba)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_val_split, y_val_proba):.4f})', color='#27ae60')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/random_forest/rf_roc_curve.png', dpi=300, bbox_inches='tight')
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
plt.savefig('../outputs/random_forest/rf_target_distribution.png', dpi=300, bbox_inches='tight')
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
bars1 = plt.bar(x - width/2, metrics_train, width, label='Train', color='#2ecc71', alpha=0.7)
bars2 = plt.bar(x + width/2, metrics_val, width, label='Validation', color='#27ae60', alpha=0.9)
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
plt.savefig('../outputs/random_forest/rf_performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Performance Metrics kaydedildi")

# 6. İlk 4 ağaç - Ayrı ayrı
for i in range(4):
    fig_tree = plt.figure(figsize=(12, 8))
    plot_tree(rf_model.estimators_[i], 
              max_depth=2,
              filled=True, 
              feature_names=X_train.columns,
              class_names=['Not Leave', 'Leave'],
              fontsize=9,
              rounded=True)
    plt.title(f'Random Forest - Ağaç #{i+1} (İlk 2 Seviye)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'../outputs/random_forest/rf_tree_{i+1}.png', dpi=300, bbox_inches='tight')
    plt.close()
print(f"  ✓ İlk 4 ağaç ayrı ayrı kaydedildi")

# Tek bir ağacın tam yapısı
print("✓ İlk ağacın tam yapısını kaydediyorum...")
fig_full = plt.figure(figsize=(25, 15))
plot_tree(rf_model.estimators_[0], 
          filled=True, 
          feature_names=X_train.columns,
          class_names=['Not Leave', 'Leave'],
          fontsize=10,
          rounded=True,
          proportion=True)
plt.title('Random Forest - İlk Ağaç (Tam Yapı)', fontsize=16, fontweight='bold', pad=20)
plt.savefig('../outputs/random_forest/random_forest_single_tree.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Tek ağaç görselleştirmesi kaydedildi")

# Ağaç derinlikleri dağılımı
fig_stats = plt.figure(figsize=(12, 6))

# Sol: Ağaç derinlikleri
ax1 = plt.subplot(1, 2, 1)
all_depths = [tree.get_depth() for tree in rf_model.estimators_]
plt.hist(all_depths, bins=range(min(all_depths), max(all_depths) + 2), 
         color='#27ae60', alpha=0.7, edgecolor='black')
plt.xlabel('Ağaç Derinliği', fontweight='bold')
plt.ylabel('Ağaç Sayısı', fontweight='bold')
plt.title('100 Ağacın Derinlik Dağılımı', fontsize=14, fontweight='bold')
plt.axvline(np.mean(all_depths), color='red', linestyle='--', 
            label=f'Ortalama: {np.mean(all_depths):.2f}')
plt.legend()
plt.grid(True, alpha=0.3)

# Sağ: Yaprak sayıları
ax2 = plt.subplot(1, 2, 2)
all_leaves = [tree.get_n_leaves() for tree in rf_model.estimators_]
plt.hist(all_leaves, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
plt.xlabel('Yaprak Sayısı', fontweight='bold')
plt.ylabel('Ağaç Sayısı', fontweight='bold')
plt.title('100 Ağacın Yaprak Sayısı Dağılımı', fontsize=14, fontweight='bold')
plt.axvline(np.mean(all_leaves), color='red', linestyle='--', 
            label=f'Ortalama: {np.mean(all_leaves):.2f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/random_forest/random_forest_tree_stats.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Ağaç istatistikleri kaydedildi")

print("\n✓ Tüm grafikler hem birleşik hem de ayrı ayrı kaydedildi!")


# ============================================================================
# 6. TEST VERİSİ İÇİN TAHMİNLER
# ============================================================================
print("\n[6] Test Verisi Tahminleri")
print("-"*70)

# Tüm train verisi ile son modeli eğit
print("⏳ Final Random Forest modeli tüm train verisi ile eğitiliyor...")
final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_split=200,
    min_samples_leaf=100,
    criterion='gini',
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    max_features='sqrt'
)
final_model.fit(X_train, y)
print("✓ Final model eğitimi tamamlandı!")

# Test tahminleri
test_predictions = final_model.predict_proba(X_test)[:, 1]

# Submission dosyasını hazırla
os.makedirs('../submissions', exist_ok=True)
submission['target'] = test_predictions
submission.to_csv('../submissions/submission_random_forest.csv', index=False)
print(f"✓ Submission dosyası oluşturuldu: submissions/submission_random_forest.csv")
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
print(f"\n✓ Model Tipi: Random Forest Classifier")
print(f"✓ Toplam Ağaç Sayısı: {final_model.n_estimators}")
print(f"✓ Her Ağaç Derinliği: {final_model.max_depth}")
print(f"✓ Ortalama Ağaç Derinliği: {np.mean([tree.get_depth() for tree in final_model.estimators_]):.2f}")
print(f"✓ Ortalama Yaprak Sayısı: {np.mean([tree.get_n_leaves() for tree in final_model.estimators_]):.2f}")
print(f"✓ Validation Accuracy: {accuracy_score(y_val_split, y_val_pred):.4f}")
print(f"✓ Validation ROC-AUC: {roc_auc_score(y_val_split, y_val_proba):.4f}")
print(f"\n📁 Oluşturulan Dosyalar:")
print(f"  Birleşik Görsel:")
print(f"    • outputs/random_forest_analysis.png - Genel analiz grafikleri")
print(f"  Ayrı Görseller:")
print(f"    • outputs/rf_confusion_matrix.png")
print(f"    • outputs/rf_feature_importance.png")
print(f"    • outputs/rf_roc_curve.png")
print(f"    • outputs/rf_target_distribution.png")
print(f"    • outputs/rf_performance_metrics.png")
print(f"    • outputs/rf_tree_1.png, rf_tree_2.png, rf_tree_3.png, rf_tree_4.png")
print(f"    • outputs/random_forest_single_tree.png - Tek ağaç tam yapısı")
print(f"    • outputs/random_forest_tree_stats.png - Ağaç istatistikleri")
print(f"  Submission:")
print(f"    • submissions/submission_random_forest.csv")

print("\n" + "="*70)
print("✅ İŞLEM TAMAMLANDI!")
print("="*70)
print("\n💡 Random Forest Özellikleri:")
print("   • 100 farklı decision tree kullanır (ensemble)")
print("   • Her ağaç farklı veri örnekleriyle eğitilir (bootstrap)")
print("   • Her dallanmada rastgele feature seçimi yapar")
print("   • Final tahmin = Tüm ağaçların ortalaması")
print("   • Tek ağaçtan daha güçlü ve robust")
print("="*70)
