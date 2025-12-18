# Employee Attrition Prediction
## Çalışan İşten Ayrılma Tahmini - Machine Learning Projesi

Bu proje, çalışanların işten ayrılma olasılığını tahmin etmek için **Decision Tree** ve **Random Forest** makine öğrenmesi modellerini kullanır ve bu iki yöntem arasındaki farkları gösterir.

---

## 📊 Veri Seti

- **aug_train.csv**: 19,158 eğitim örneği
- **aug_test.csv**: 2,129 test örneği
- **Özellikler**: 12 feature (şehir, deneyim, eğitim, şirket bilgileri vb.)
- **Hedef**: Binary sınıflandırma (0: Kalmaya devam, 1: İşten ayrılacak)

### Veri Özellikleri:
- `city`: Şehir kodu
- `city_development_index`: Şehir gelişmişlik endeksi
- `gender`: Cinsiyet
- `relevent_experience`: İlgili deneyim durumu
- `enrolled_university`: Üniversite kayıt durumu
- `education_level`: Eğitim seviyesi
- `major_discipline`: Ana disiplin
- `experience`: Toplam deneyim yılı
- `company_size`: Şirket büyüklüğü
- `company_type`: Şirket tipi
- `last_new_job`: Son iş değişikliği
- `training_hours`: Eğitim saatleri

---

## 🚀 Kullanım

### 1. Decision Tree Modeli
```bash
python src/decision_tree_model.py
```

**Çıktılar:**
- `outputs/decision_tree/decision_tree_analysis.png` - Birleşik analiz grafikleri
- `outputs/decision_tree/decision_tree_full.png` - Tam ağaç yapısı
- `outputs/decision_tree/dt_confusion_matrix.png` - Confusion matrix
- `outputs/decision_tree/dt_feature_importance.png` - Özellik önemleri
- `outputs/decision_tree/dt_roc_curve.png` - ROC eğrisi
- `submissions/submission_decision_tree.csv` - Test tahminleri

### 2. Random Forest Modeli
```bash
python src/random_forest_model.py
```

**Çıktılar:**
- `outputs/random_forest/random_forest_analysis.png` - Birleşik analiz grafikleri (4 ağaç örneği)
- `outputs/random_forest/random_forest_single_tree.png` - Tek ağaç tam yapısı
- `outputs/random_forest/random_forest_tree_stats.png` - Ağaç istatistikleri
- `outputs/random_forest/rf_confusion_matrix.png` - Confusion matrix
- `outputs/random_forest/rf_feature_importance.png` - Özellik önemleri
- `outputs/random_forest/rf_roc_curve.png` - ROC eğrisi
- `submissions/submission_random_forest.csv` - Test tahminleri

**Not:** Bu model Label Encoding kullanır

### 3. Model Karşılaştırması (Decision Tree vs Random Forest)
```bash
python src/compare_models.py
```

**Çıktılar:**
- `outputs/compare_models/model_comparison.png` - Birleşik karşılaştırma grafikleri
- `outputs/compare_models/compare_metrics.png` - Metrik karşılaştırması
- `outputs/compare_models/compare_roc_curves.png` - ROC eğrileri karşılaştırması
- `outputs/compare_models/compare_dt_confusion_matrix.png` - Decision Tree confusion matrix
- `outputs/compare_models/compare_rf_confusion_matrix.png` - Random Forest confusion matrix
- `outputs/compare_models/compare_feature_importance.png` - Özellik önemleri karşılaştırması
- `outputs/compare_models/compare_overfitting.png` - Overfitting analizi

**Not:** Bu karşılaştırma One-Hot Encoding kullanır

---

## 📈 Model Performansları

### Decision Tree (Validation Set)
| Metrik | Değer |
|--------|-------|
| **Accuracy** | 0.6962 |
| **Precision** | 0.4393 |
| **Recall** | 0.7916 |
| **F1-Score** | 0.5650 |
| **ROC-AUC** | 0.7816 |

**Model Özellikleri:**
- Ağaç Derinliği: 5
- Yaprak Sayısı: 26
- Tek ağaç kullanır
- Basit ve yorumlanabilir

### Random Forest (Validation Set)

#### Label Encoding ile:
| Metrik | Değer |
|--------|-------|
| **Accuracy** | 0.7677 |
| **Precision** | 0.5272 |
| **Recall** | 0.6586 |
| **F1-Score** | 0.5857 |
| **ROC-AUC** | 0.7877 |

#### One-Hot Encoding ile (Compare Models):
| Metrik | Değer |
|--------|-------|
| **Accuracy** | 0.7523 |
| **Precision** | 0.5024 |
| **Recall** | 0.6639 |
| **F1-Score** | 0.5719 |
| **ROC-AUC** | 0.7808 |

**Model Özellikleri:**
- Ağaç Sayısı: 100
- Her ağaç derinliği: 4
- Ortalama yaprak sayısı: ~14.21
- Ensemble metodu
- Daha robust ve dengeli
- **En iyi sonuç**: Label Encoding ile

### İyileşmeler (Random Forest vs Decision Tree)

#### Label Encoding ile:
- **Accuracy**: +10.27% ↑
- **Precision**: +20.01% ↑
- **Recall**: -16.81% ↓
- **F1-Score**: +3.66% ↑
- **ROC-AUC**: +0.78% ↑

#### One-Hot Encoding ile (Compare Models):
- **Accuracy**: +8.06% ↑
- **Precision**: +14.36% ↑
- **Recall**: -16.14% ↓
- **F1-Score**: +1.23% ↑
- **ROC-AUC**: -0.11% ↓

---

## 🌳 Decision Tree vs 🌲 Random Forest

### Decision Tree
✅ **Avantajlar:**
- Yorumlanabilir ve anlaşılır
- Hızlı eğitim
- Tek model, basit
- Görselleştirilebilir

❌ **Dezavantajlar:**
- Overfitting riski yüksek
- Küçük veri değişikliklerinde kararsız
- Düşük genelleme

### Random Forest
✅ **Avantajlar:**
- Yüksek doğruluk
- Overfitting riski düşük
- Robust ve kararlı
- Feature importance güvenilir

❌ **Dezavantajlar:**
- Yorumlanması zor
- Yavaş eğitim
- Daha fazla kaynak gerekir
- Black-box model

---

## 🔍 Temel Farklar

| Özellik | Decision Tree | Random Forest |
|---------|---------------|---------------|
| Ağaç Sayısı | 1 | 100 |
| Ağaç Derinliği | 5 | 4 (her biri) |
| Yaprak Sayısı | 26 | ~14.21 (her ağaç) |
| Veri Örnekleme | Tüm veri | Bootstrap sampling |
| Feature Seçimi | Tüm features | Rastgele subset (sqrt) |
| Tahmin | Tek ağaç | Ağaçların ortalaması |
| Yorumlanabilirlik | Yüksek | Düşük |
| Accuracy | 69.62% | 76.77% |

---

## 📊 En Önemli Özellikler

### Decision Tree
1. **city_development_index** (0.6045) - Şehir gelişmişlik endeksi
2. **company_size_50-99** (0.2238) - Şirket büyüklüğü (50-99 çalışan)
3. **education_level_Graduate** (0.0519) - Eğitim seviyesi (Lisans)
4. **relevent_experience** (0.0500) - İlgili deneyim
5. **city_city_103** (0.0209) - Şehir 103

### Random Forest
1. **city_development_index** (0.5407) - Şehir gelişmişlik endeksi
2. **city** (0.1287) - Şehir kodu
3. **company_size** (0.1045) - Şirket büyüklüğü
4. **enrolled_university** (0.0755) - Üniversite kayıt durumu
5. **relevent_experience** (0.0624) - İlgili deneyim

---

## 🛠️ Teknolojiler

- **Python 3.13**
- **pandas** - Veri manipülasyonu
- **numpy** - Sayısal hesaplamalar
- **scikit-learn** - Machine learning modelleri
- **matplotlib** - Görselleştirme
- **seaborn** - İstatistiksel görselleştirme

---

## 📝 Veri Ön İşleme Adımları

1. **Eksik Değer Doldurma:**
   - Kategorik değişkenler → Mode (en sık görülen değer)

2. **Encoding:**
   - **Decision Tree**: One-Hot Encoding (186 feature)
   - **Random Forest**: Label Encoding (12 feature)
   - **Compare Models**: Her iki model için One-Hot Encoding

3. **Train-Validation Split:**
   - 80% Train, 20% Validation
   - Stratified split (sınıf dengesi korundu)

4. **Class Balancing:**
   - `class_weight='balanced'` parametresi kullanıldı

---

## 📉 Model Hiperparametreleri

### Decision Tree
```python
DecisionTreeClassifier(
    max_depth=5,              # Ağaç derinliği (5 seviye)
    min_samples_split=100,    # Dallanma için min örnek
    min_samples_leaf=50,      # Her yaprakta min örnek
    criterion='gini',         # Bölünme kriteri
    class_weight='balanced'   # Sınıf dengesi
)
```

### Random Forest
```python
RandomForestClassifier(
    n_estimators=100,         # 100 ağaç
    max_depth=4,              # Her ağaç için max derinlik
    min_samples_split=200,
    min_samples_leaf=100,
    criterion='gini',
    class_weight='balanced',
    n_jobs=-1                 # Paralel işleme
)
```

---

## 📂 Proje Yapısı

```
employee-attrition/
├── data/
│   ├── aug_train.csv                # Eğitim verisi
│   ├── aug_test.csv                 # Test verisi
│   └── sample_submission.csv        # Örnek submission formatı
├── src/
│   ├── decision_tree_model.py       # Decision Tree modeli
│   ├── random_forest_model.py       # Random Forest modeli
│   ├── compare_models.py            # Model karşılaştırması
│   └── model_builders.py            # Yardımcı fonksiyonlar
├── outputs/
│   ├── decision_tree/               # DT çıktıları
│   │   ├── decision_tree_analysis.png
│   │   ├── decision_tree_full.png
│   │   └── ... (diğer grafikler)
│   ├── random_forest/               # RF çıktıları
│   │   ├── random_forest_analysis.png
│   │   ├── random_forest_single_tree.png
│   │   └── ... (diğer grafikler)
│   └── compare_models/              # Karşılaştırma çıktıları
│       ├── model_comparison.png
│       └── ... (diğer grafikler)
├── submissions/
│   ├── submission_decision_tree.csv # DT tahminleri
│   └── submission_random_forest.csv # RF tahminleri
└── README.md                        # Bu dosya
```

---

## 🎯 Sonuçlar ve Öneriler

### Sonuçlar:
1. **Random Forest (Label Encoding)** en yüksek accuracy (%76.77) sağladı
2. **Decision Tree** en yüksek recall (%79.16) gösterdi - daha fazla attrition vakasını yakaladı
3. **Random Forest** daha dengeli performans sundu (precision ve recall dengeli)
4. Overfitting, Random Forest'ta daha az görüldü
5. **Decision Tree** tek ağaçla %69.62 accuracy elde etti
6. **Label Encoding** Random Forest için One-Hot Encoding'den daha iyi sonuç verdi
7. **Compare Models** sonuçları: RF %75.23 vs DT %69.62 (One-Hot Encoding ile)

### Öneriler:
- **Üretim için**: Random Forest (daha güvenilir)
- **Açıklama gerekiyorsa**: Decision Tree (yorumlanabilir)
- **Hızlı prototip**: Decision Tree (daha hızlı)
- **En iyi performans**: Random Forest veya Gradient Boosting

---

## 🔮 Gelecek Geliştirmeler

- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Feature engineering
- [ ] SMOTE ile class balancing
- [ ] Gradient Boosting modelleri (XGBoost, LightGBM)
- [ ] Cross-validation
- [ ] Feature selection
- [ ] Ensemble of ensembles

---

## 📧 İletişim

Bu proje, Decision Tree ve Random Forest arasındaki farkları göstermek amacıyla oluşturulmuştur.

**Tarih:** Aralık 2025

---

## 📜 Lisans

Bu proje eğitim amaçlıdır.
