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
python decision_tree_model.py
```

**Çıktılar:**
- `decision_tree_analysis.png` - Genel analiz grafikleri
- `decision_tree_full.png` - Tam ağaç yapısı
- `submission_decision_tree.csv` - Test tahminleri

### 2. Model Karşılaştırması (Decision Tree vs Random Forest)
```bash
python compare_models.py
```

**Çıktılar:**
- `model_comparison.png` - Karşılaştırma grafikleri
- `submission_random_forest.csv` - Random Forest tahminleri

---

## 📈 Model Performansları

### Decision Tree (Validation Set)
| Metrik | Değer |
|--------|-------|
| **Accuracy** | 0.6806 |
| **Precision** | 0.4253 |
| **Recall** | 0.8021 |
| **F1-Score** | 0.5559 |
| **ROC-AUC** | 0.7823 |

**Model Özellikleri:**
- Ağaç Derinliği: 6
- Yaprak Sayısı: 40
- Tek ağaç kullanır
- Yorumlanabilir

### Random Forest (Validation Set)
| Metrik | Değer |
|--------|-------|
| **Accuracy** | 0.7657 |
| **Precision** | 0.5231 |
| **Recall** | 0.6754 |
| **F1-Score** | 0.5896 |
| **ROC-AUC** | 0.7956 |

**Model Özellikleri:**
- Ağaç Sayısı: 100
- Her ağaç derinliği: 6
- Ensemble metodu
- Daha robust

### İyileşmeler (Random Forest)
- **Accuracy**: +12.50% ↑
- **Precision**: +22.99% ↑
- **Recall**: -15.80% ↓
- **F1-Score**: +6.06% ↑
- **ROC-AUC**: +1.70% ↑

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
| Veri Örnekleme | Tüm veri | Bootstrap sampling |
| Feature Seçimi | Tüm features | Rastgele subset |
| Tahmin | Tek ağaç | Ağaçların ortalaması |
| Yorumlanabilirlik | Yüksek | Düşük |
| Doğruluk | Düşük | Yüksek |

---

## 📊 En Önemli Özellikler

1. **city_development_index** (0.5853) - Şehir gelişmişlik endeksi
2. **company_size** (0.2269) - Şirket büyüklüğü
3. **education_level** (0.0511) - Eğitim seviyesi
4. **relevent_experience** (0.0511) - İlgili deneyim
5. **city** (0.0351) - Şehir

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
   - Numerik değişkenler → Median
   - Kategorik değişkenler → Mode (en sık görülen)

2. **Encoding:**
   - Label Encoding (tüm kategorik değişkenler için)

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
    max_depth=6,              # Çok dallı olmasın
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
    max_depth=6,              # Her ağaç için max derinlik
    min_samples_split=100,
    min_samples_leaf=50,
    criterion='gini',
    class_weight='balanced',
    n_jobs=-1                 # Paralel işleme
)
```

---

## 📂 Proje Yapısı

```
employee-attrition/
├── aug_train.csv                      # Eğitim verisi
├── aug_test.csv                       # Test verisi
├── sample_submission.csv              # Submission şablonu
├── decision_tree_model.py             # Decision Tree modeli
├── compare_models.py                  # Model karşılaştırması
├── decision_tree_analysis.png         # DT analiz grafikleri
├── decision_tree_full.png             # Tam ağaç yapısı
├── model_comparison.png               # Karşılaştırma grafikleri
├── submission_decision_tree.csv       # DT tahminleri
├── submission_random_forest.csv       # RF tahminleri
└── README.md                          # Bu dosya
```

---

## 🎯 Sonuçlar ve Öneriler

### Sonuçlar:
1. **Random Forest** daha yüksek accuracy (%76.6) sağladı
2. **Decision Tree** daha yüksek recall (%80.2) gösterdi
3. **Random Forest** daha dengeli performans sundu
4. Overfitting, Random Forest'ta daha az

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
