"""
Employee Attrition Prediction - Decision Tree Model
Çalışan işten ayrılma tahmini için Decision Tree modeli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
print("EMPLOYEE ATTRITION PREDICTION - DECISION TREE MODEL")
print("="*70)

# ============================================================================
# 1. VERİ YÜKLEME VE KEŞİF ANALİZİ
# ============================================================================
print("\n[1] Veri Yükleme ve Keşif Analizi")
print("-"*70)

# Veri setlerini yükle
train_df = pd.read_csv('aug_train.csv')
test_df = pd.read_csv('aug_test.csv')
submission = pd.read_csv('sample_submission.csv')

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

# enrollee_id'yi ayır (model için kullanılmayacak)
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

# Kategorik sütunlar için mode (en sık görülen değer)
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

# Decision Tree modeli - Basit ve az dallı (Random Forest ile karşılaştırma için)
print("\n🌳 Decision Tree parametreleri:")
print("  - max_depth: 4 (ağacın maksimum derinliği - basit tutuldu)")
print("  - min_samples_split: 200 (dallanma için minimum örnek sayısı)")
print("  - min_samples_leaf: 100 (yaprak düğümdeki minimum örnek sayısı)")
print("  - criterion: gini (bölünme kriteri)")
print("  - random_state: 42")

dt_model = DecisionTreeClassifier(
    max_depth=4,                    # Basit ve anlaşılır ağaç için düşük derinlik
    min_samples_split=200,          # Daha az dallanma için artırıldı
    min_samples_leaf=100,           # Her yaprakta daha fazla örnek - daha az dal
    criterion='gini',               # Gini impurity kullan
    random_state=42,
    class_weight='balanced'         # Dengesiz veri için sınıf ağırlıkları
)

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
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']:30s} : {row['Importance']:.4f}")

# ============================================================================
# 5. GÖRSELLEŞTİRME
# ============================================================================
print("\n[5] Görselleştirmeler Oluşturuluyor...")
print("-"*70)

# Figure oluştur
fig = plt.figure(figsize=(20, 12))

# 1. Confusion Matrix
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Leave', 'Leave'],
            yticklabels=['Not Leave', 'Leave'])
plt.title('Confusion Matrix (Validation)', fontsize=14, fontweight='bold')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin')

# 2. Feature Importance
ax2 = plt.subplot(2, 3, 2)
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 10 Önemli Özellikler', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# 3. ROC Curve
ax3 = plt.subplot(2, 3, 3)
fpr, tpr, _ = roc_curve(y_val_split, y_val_proba)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_val_split, y_val_proba):.4f})')
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
plt.bar(x - width/2, metrics_train, width, label='Train', color='#2ecc71')
plt.bar(x + width/2, metrics_val, width, label='Validation', color='#3498db')
plt.xlabel('Metrikler')
plt.ylabel('Skor')
plt.title('Model Performansı Karşılaştırması', fontsize=14, fontweight='bold')
plt.xticks(x, ['Accuracy', 'Precision', 'Recall', 'F1-Score'], rotation=45)
plt.legend()
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')

# 6. Decision Tree yapısını göster (basitleştirilmiş)
ax6 = plt.subplot(2, 3, 6)
plot_tree(dt_model, 
          max_depth=2,  # Görselleştirme için sadece ilk 2 seviye
          filled=True, 
          feature_names=X_train.columns,
          class_names=['Not Leave', 'Leave'],
          fontsize=8,
          rounded=True)
plt.title('Decision Tree Yapısı (İlk 2 Seviye)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('decision_tree_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Grafik kaydedildi: decision_tree_analysis.png")

# Daha detaylı ağaç görselleştirmesi
fig2 = plt.figure(figsize=(25, 15))
plot_tree(dt_model, 
          filled=True, 
          feature_names=X_train.columns,
          class_names=['Not Leave', 'Leave'],
          fontsize=10,
          rounded=True,
          proportion=True)
plt.title('Decision Tree - Tam Yapı', fontsize=16, fontweight='bold', pad=20)
plt.savefig('decision_tree_full.png', dpi=300, bbox_inches='tight')
print("✓ Tam ağaç görselleştirmesi kaydedildi: decision_tree_full.png")

# ============================================================================
# 6. TEST VERİSİ İÇİN TAHMİNLER
# ============================================================================
print("\n[6] Test Verisi Tahminleri")
print("-"*70)

# Tüm train verisi ile son modeli eğit
print("⏳ Final model tüm train verisi ile eğitiliyor...")
final_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=200,
    min_samples_leaf=100,
    criterion='gini',
    random_state=42,
    class_weight='balanced'
)
final_model.fit(X_train, y)
print("✓ Final model eğitimi tamamlandı!")

# Test tahminleri
test_predictions = final_model.predict_proba(X_test)[:, 1]

# Submission dosyasını hazırla
submission['target'] = test_predictions
submission.to_csv('submission_decision_tree.csv', index=False)
print(f"✓ Submission dosyası oluşturuldu: submission_decision_tree.csv")
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
print(f"  • decision_tree_analysis.png - Genel analiz grafikleri")
print(f"  • decision_tree_full.png - Tam ağaç yapısı")
print(f"  • submission_decision_tree.csv - Test tahminleri")

print("\n" + "="*70)
print("✅ İŞLEM TAMAMLANDI!")
print("="*70)
print("\n💡 Not: Decision Tree basit ve yorumlanabilir bir modeldir.")
print("   Random Forest ile karşılaştırma yapmak için birden fazla")
print("   ağacın ensemble'ını kullanmanız gerekecek.")
print("="*70)
