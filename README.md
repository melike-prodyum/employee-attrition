# Employee Attrition Prediction
## Çalışan İşten Ayrılma Tahmini - Machine Learning Projesi

Bu proje, çalışanların işten ayrılma olasılığını tahmin etmek için **Decision Tree** ve **Random Forest** makine öğrenmesi modellerini kullanır. Proje, modüler kod yapısı ile geliştirilmiş olup, iki yöntem arasındaki farkları detaylı analiz ve görselleştirmelerle gösterir.

### 🎯 Proje Özellikleri
- ✅ Modüler ve yeniden kullanılabilir kod yapısı
- ✅ Detaylı görselleştirmeler ve analizler
- ✅ Decision Tree ve Random Forest karşılaştırması
- ✅ Kapsamlı model değerlendirme metrikleri
- ✅ One-Hot Encoding ile özellik mühendisliği
- ✅ Class balancing ile imbalanced dataset yönetimi
- ✅ Submission dosyaları üretimi

---

## 🚀 Kurulum ve Başlangıç

### Gereksinimler
- Python 3.8 veya üzeri
- pip package manager

### 1. Projeyi Klonlayın
```bash
git clone <repository-url>
cd employee-attrition
```

### 2. Virtual Environment Oluşturun
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Gerekli Paketleri Yükleyin
```bash
pip install -r requirements.txt
```

**Gerekli Paketler:**
- pandas (>= 1.5.0)
- numpy (>= 1.23.0)
- scikit-learn (>= 1.2.0)
- matplotlib (>= 3.6.0)
- seaborn (>= 0.12.0)

### 4. Veri Setlerini Hazırlayın
`data/` klasöründe aşağıdaki dosyaların bulunduğundan emin olun:
- `aug_train.csv`
- `aug_test.csv`
- `sample_submission.csv`

### 5. Hızlı Başlangıç
```bash
# Tek bir model çalıştırmak için:
python src/decision_tree_model.py

# Veya Random Forest:
python src/random_forest_model.py

# Karşılaştırma yapmak için:
python src/compare_models.py
```

---

## 📊 Veri Seti

- **aug_train.csv**: 19,158 eğitim örneği
- **aug_test.csv**: 2,129 test örneği
- **Özellikler**: 12 feature (şehir, deneyim, eğitim, şirket bilgileri vb.)
- **Hedef**: Binary sınıflandırma (0: Kalmaya devam, 1: İşten ayrılacak)

### Veri Özellikleri:
- `enrollee_id`: Çalışan kimlik numarası
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
- `target`: Hedef değişken (0/1)

---

## 🚀 Modelleri Çalıştırma

Projenin kök dizininde (README.md ile aynı klasörde), virtual environment aktifken aşağıdaki komutları çalıştırın:

### 1. Decision Tree Modeli
```bash
python src/decision_tree_model.py
```

**Özellikler:**
- One-Hot Encoding kullanır
- Validation set ile model performansını değerlendirir
- Detaylı görselleştirmeler oluşturur

**Çıktılar:**
- `outputs/decision_tree/decision_tree_analysis.png` - Birleşik analiz grafikleri (4'lü panel)
- `outputs/decision_tree/decision_tree_full.png` - Tam ağaç yapısı görselleştirmesi
- `outputs/decision_tree/dt_confusion_matrix.png` - Confusion matrix
- `outputs/decision_tree/dt_feature_importance.png` - Özellik önemleri
- `outputs/decision_tree/dt_roc_curve.png` - ROC eğrisi
- `submissions/submission_decision_tree.csv` - Test tahminleri

### 2. Random Forest Modeli
```bash
python src/random_forest_model.py
```

**Özellikler:**
- One-Hot Encoding kullanır (gelişmiş performans için)
- 100 ağaçlı ensemble model
- Her ağaç için detaylı istatistikler

**Çıktılar:**
- `outputs/random_forest/random_forest_analysis.png` - Birleşik analiz (4 farklı ağaç örneği)
- `outputs/random_forest/random_forest_single_tree.png` - Tek ağacın tam yapısı
- `outputs/random_forest/random_forest_tree_stats.png` - Ağaç istatistikleri histogramı
- `outputs/random_forest/rf_confusion_matrix.png` - Confusion matrix
- `outputs/random_forest/rf_feature_importance.png` - Özellik önemleri
- `outputs/random_forest/rf_roc_curve.png` - ROC eğrisi
- `submissions/submission_random_forest.csv` - Test tahminleri

### 3. Model Karşılaştırması
```bash
python src/compare_models.py
```

**Özellikler:**
- Her iki modeli aynı veri üzerinde karşılaştırır
- One-Hot Encoding ile adil karşılaştırma
- Detaylı performans analizi ve overfitting karşılaştırması

**Çıktılar:**
- `outputs/compare_models/model_comparison.png` - Birleşik karşılaştırma (6'lı panel)
- `outputs/compare_models/compare_metrics.png` - Metrik karşılaştırması bar grafikleri
- `outputs/compare_models/compare_roc_curves.png` - ROC eğrileri üst üste
- `outputs/compare_models/compare_dt_confusion_matrix.png` - Decision Tree confusion matrix
- `outputs/compare_models/compare_rf_confusion_matrix.png` - Random Forest confusion matrix
- `outputs/compare_models/compare_feature_importance.png` - Özellik önemleri karşılaştırması
- `outputs/compare_models/compare_overfitting.png` - Overfitting analizi (train vs validation)

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
- Tek ağaç yapısı
- Maksimum derinlik: 5 seviye
- Basit ve yorumlanabilir
- Görselleştirilebilir karar yapısı

### Random Forest (Validation Set)
| Metrik | Değer |
|--------|-------|
| **Accuracy** | 0.7523 |
| **Precision** | 0.5024 |
| **Recall** | 0.6639 |
| **F1-Score** | 0.5719 |
| **ROC-AUC** | 0.7808 |

**Model Özellikleri:**
- 100 ağaçlı ensemble yapı
- Her ağaç maksimum derinlik: 3 seviye
- Bootstrap sampling ile veri çeşitliliği
- Rastgele özellik seçimi (sqrt)
- Daha robust ve kararlı tahminler

### İyileşmeler (Random Forest vs Decision Tree)
- **Accuracy**: +8.06% ↑ (0.6962 → 0.7523)
- **Precision**: +14.36% ↑ (0.4393 → 0.5024)
- **Recall**: -16.14% ↓ (0.7916 → 0.6639)
- **F1-Score**: +1.23% ↑ (0.5650 → 0.5719)
- **ROC-AUC**: -0.11% ↓ (0.7816 → 0.7808)

---

## 🌳 Decision Tree vs 🌲 Random Forest

### Decision Tree
✅ **Avantajlar:**
- Yorumlanabilir ve anlaşılır karar yapısı
- Hızlı eğitim ve tahmin
- Tek model ile basit implementasyon
- Görselleştirme ile karar sürecini gösterebilir
- Az bellek tüketimi

❌ **Dezavantajlar:**
- Yüksek overfitting riski
- Veri değişikliklerine karşı hassas
- Düşük genelleme yeteneği
- Yüksek varyans

### Random Forest
✅ **Avantajlar:**
- Yüksek doğruluk ve performans
- Overfitting riski çok düşük
- Robust ve kararlı sonuçlar
- Güvenilir feature importance
- Outlier'lara karşı dirençli
- Varyansı düşürür

❌ **Dezavantajlar:**
- Black-box model (yorumlama zor)
- Yavaş eğitim süresi
- Daha fazla bellek gerektirir
- Bireysel ağaçları görselleştirmek zor
- Daha fazla hesaplama kaynağı gerekir

---

## 🔍 Temel Farklar

| Özellik | Decision Tree | Random Forest |
|---------|---------------|---------------|
| **Ağaç Sayısı** | 1 | 100 |
| **Ağaç Derinliği** | 5 | 3 (her biri) |
| **Veri Örnekleme** | Tüm veri | Bootstrap sampling |
| **Feature Seçimi** | Tüm features | Rastgele subset (sqrt) |
| **Tahmin Yöntemi** | Tek ağaç çıktısı | Ağaçların oylama ortalaması |
| **Yorumlanabilirlik** | Yüksek | Düşük |
| **Eğitim Süresi** | Hızlı | Yavaş |
| **Overfitting** | Yüksek risk | Düşük risk |
| **Accuracy** | 69.62% | 75.23% |
| **Encoding** | One-Hot | One-Hot |

---

## 📊 En Önemli Özellikler

### Decision Tree (One-Hot Encoding)
1. **city_development_index** (0.6045) - Şehir gelişmişlik endeksi
2. **company_size_50-99** (0.2238) - Orta ölçekli şirket
3. **education_level_Graduate** (0.0519) - Lisans mezunu
4. **relevent_experience** (0.0500) - İlgili deneyim durumu
5. **city_city_103** (0.0209) - Belirli şehir

### Random Forest (One-Hot Encoding)
1. **city_development_index** - Şehir gelişmişlik endeksi (en baskın özellik)
2. **company_size features** - Şirket büyüklüğü kategorileri
3. **education_level features** - Eğitim seviyesi kategorileri
4. **experience features** - Deneyim yılı kategorileri
5. **enrolled_university features** - Üniversite kayıt durumu

*Not: Random Forest'ta özellikler ensemble genelinde agregedir, bu nedenle bireysel feature importance'lar daha dengeli dağılır.*

---

## 🛠️ Teknolojiler

- **Python 3.x**
- **pandas** - Veri manipülasyonu ve analizi
- **numpy** - Sayısal hesaplamalar
- **scikit-learn** - Machine learning modelleri ve metrikler
- **matplotlib** - Temel görselleştirme
- **seaborn** - İstatistiksel görselleştirme

---

## 📝 Veri Ön İşleme Pipeline

### 1. Veri Yükleme
- `load_data()` fonksiyonu ile train, test ve submission template yüklenir

### 2. Eksik Değer Doldurma
- **Kategorik değişkenler**: Mode (en sık görülen değer) ile doldurulur
- **Numerik değişkenler**: Median ile doldurulur
- Train ve test setleri aynı değerlerle doldurulur (data leakage önlenir)

### 3. Feature Encoding
- **Decision Tree**: One-Hot Encoding
- **Random Forest**: One-Hot Encoding
- **Compare Models**: Her iki model için One-Hot Encoding

### 4. Train-Validation Split
- **Oran**: 80% Train, 20% Validation
- **Stratified Split**: Sınıf dengesi korunur
- **Random State**: 42 (reproducibility için)

### 5. Class Balancing
- `class_weight='balanced'` parametresi ile azınlık sınıfına daha fazla ağırlık verilir
- İmbalanced dataset problemi çözülür

---

## 📉 Model Hiperparametreleri

### Decision Tree
```python
DecisionTreeClassifier(
    max_depth=5,                # Ağaç derinliği (overfitting kontrolü)
    min_samples_split=100,      # Dallanma için minimum örnek sayısı
    min_samples_leaf=50,        # Her yaprakta minimum örnek
    criterion='gini',           # Gini impurity ile bölünme
    random_state=42,            # Reproducibility
    class_weight='balanced'     # Otomatik sınıf ağırlıklandırma
)
```

### Random Forest
```python
RandomForestClassifier(
    n_estimators=100,           # 100 farklı decision tree
    max_depth=3,                # Her ağaç için derinlik (basit ağaçlar)
    min_samples_split=300,      # Dallanma için minimum örnek
    min_samples_leaf=150,       # Her yaprakta minimum örnek
    criterion='gini',           # Gini impurity
    random_state=42,            # Reproducibility
    class_weight='balanced',    # Sınıf dengesi
    n_jobs=-1,                  # Tüm CPU çekirdekleri kullanılır
    max_features='sqrt'         # Her bölünmede sqrt(n_features) özellik
)
```

---

## 📂 Proje Yapısı

```
employee-attrition/
├── .venv/                              # Python virtual environment
├── .git/                               # Git repository
├── .gitignore                          # Git ignore dosyası
│
├── data/
│   ├── aug_train.csv                   # Eğitim verisi (19,158 örnek)
│   ├── aug_test.csv                    # Test verisi (2,129 örnek)
│   └── sample_submission.csv           # Örnek submission formatı
│
├── src/
│   ├── decision_tree_model.py          # Decision Tree ana script
│   ├── random_forest_model.py          # Random Forest ana script
│   ├── compare_models.py               # Model karşılaştırma script
│   ├── model_builders.py               # Model builder fonksiyonları
│   ├── data_utils.py                   # Veri işleme utility fonksiyonları
│   ├── evaluation_utils.py             # Değerlendirme utility fonksiyonları
│   └── __pycache__/                    # Python cache dosyaları
│
├── outputs/
│   ├── decision_tree/                  # Decision Tree çıktıları
│   │   ├── decision_tree_analysis.png
│   │   ├── decision_tree_full.png
│   │   ├── dt_confusion_matrix.png
│   │   ├── dt_feature_importance.png
│   │   └── dt_roc_curve.png
│   │
│   ├── random_forest/                  # Random Forest çıktıları
│   │   ├── random_forest_analysis.png
│   │   ├── random_forest_single_tree.png
│   │   ├── random_forest_tree_stats.png
│   │   ├── rf_confusion_matrix.png
│   │   ├── rf_feature_importance.png
│   │   └── rf_roc_curve.png
│   │
│   └── compare_models/                 # Karşılaştırma çıktıları
│       ├── model_comparison.png
│       ├── compare_metrics.png
│       ├── compare_roc_curves.png
│       ├── compare_dt_confusion_matrix.png
│       ├── compare_rf_confusion_matrix.png
│       ├── compare_feature_importance.png
│       └── compare_overfitting.png
│
├── submissions/
│   ├── submission_decision_tree.csv    # Decision Tree test tahminleri
│   └── submission_random_forest.csv    # Random Forest test tahminleri
│
├── docs/                               # Dokümantasyon (opsiyonel)
├── requirements.txt                    # Python paket gereksinimleri
└── README.md                           # Bu dosya
```

---

## 🔧 Modüler Kod Yapısı

### `data_utils.py`
Veri işleme için ortak fonksiyonlar:
- `load_data()` - Veri setlerini yükler
- `print_data_info()` - Veri seti bilgilerini gösterir
- `prepare_features()` - Features ve target'ı ayırır
- `get_column_types()` - Kategorik ve numerik sütunları belirler
- `fill_missing_values()` - Eksik değerleri doldurur
- `apply_one_hot_encoding()` - One-Hot Encoding uygular
- `apply_label_encoding()` - Label Encoding uygular
- `create_output_directory()` - Output klasörü oluşturur
- `create_submission_file()` - Submission dosyası oluşturur

### `model_builders.py`
Model oluşturma fonksiyonları:
- `build_decision_tree()` - Decision Tree modeli oluşturur
- `build_random_forest()` - Random Forest modeli oluşturur
- `get_decision_tree_params()` - DT parametrelerini döndürür
- `get_random_forest_params()` - RF parametrelerini döndürür

### `evaluation_utils.py`
Model değerlendirme fonksiyonları:
- `calculate_metrics()` - Performans metriklerini hesaplar
- `print_metrics()` - Metrikleri yazdırır
- `print_classification_report()` - Detaylı rapor yazdırır
- `print_confusion_matrix()` - Confusion matrix yazdırır
- `print_feature_importance()` - Feature importance yazdırır

---

## 🎯 Sonuçlar ve Öneriler

### Sonuçlar:
1. **Random Forest** %75.23 accuracy ile Decision Tree'den (%69.62) daha iyi performans gösterdi
2. **Decision Tree** %79.16 recall ile attrition vakalarını yakalamada daha agresif
3. **Random Forest** daha dengeli precision-recall dengesine sahip
4. Overfitting analizi Random Forest'ın daha generalize edebilen bir model olduğunu gösterdi
5. **city_development_index** her iki modelde de en önemli özellik
6. Random Forest'ta özellik önemleri daha dengeli dağılım gösteriyor
7. One-Hot Encoding her iki model için kullanıldığında adil karşılaştırma yapılabiliyor

### Öneriler:
- **Üretim ortamı için**: Random Forest (daha güvenilir ve robust)
- **Açıklanabilirlik gerekiyorsa**: Decision Tree (kolay yorumlanabilir)
- **Hızlı prototipleme**: Decision Tree (hızlı eğitim ve test)
- **En yüksek performans**: Random Forest veya Gradient Boosting denenebilir
- **İmbalanced dataset**: class_weight='balanced' kullanımı önemli

---

## 🔮 Gelecek Geliştirmeler

- [ ] **Hyperparameter Tuning**: GridSearchCV veya RandomizedSearchCV ile optimal parametreler
- [ ] **Feature Engineering**: Yeni özellikler türetme (interaction features, polynomial features)
- [ ] **Advanced Resampling**: SMOTE, ADASYN ile class balancing
- [ ] **Ensemble Methods**: Voting, Stacking ile model kombinasyonu
- [ ] **Gradient Boosting**: XGBoost, LightGBM, CatBoost modelleri
- [ ] **Cross-Validation**: K-Fold CV ile daha robust değerlendirme
- [ ] **Feature Selection**: SelectKBest, RFE ile özellik seçimi
- [ ] **Deep Learning**: Neural Network modelleri deneme
- [ ] **Explainability**: SHAP, LIME ile model açıklanabilirliği
- [ ] **API Development**: Flask/FastAPI ile model servisi
- [ ] **Dockerization**: Docker container ile deployment

---

## 🔧 Sorun Giderme

### Yaygın Hatalar ve Çözümleri

**1. ModuleNotFoundError:**
```bash
# Çözüm: Gerekli paketleri yükleyin
pip install -r requirements.txt
```

**2. FileNotFoundError (veri bulunamadı):**
```bash
# Çözüm: Projenin kök dizininden çalıştırdığınızdan emin olun
cd c:\Users\botyum\source\repos\employee-attrition
python src/decision_tree_model.py
```

**3. Virtual environment aktif değil:**
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

**4. Görselleştirmeler açılmıyor:**
- Matplotlib backend'ini kontrol edin
- `outputs/` klasöründeki PNG dosyalarını manuel olarak açın

---

## � Kullanım İpuçları

### Önerilen Çalışma Akışı
1. **İlk Çalıştırma**: `python src/decision_tree_model.py` ile başlayın
2. **Karşılaştırma**: `python src/random_forest_model.py` çalıştırın
3. **Analiz**: `python src/compare_models.py` ile detaylı karşılaştırma yapın
4. **Görselleştirmeler**: `outputs/` klasöründeki grafikleri inceleyin
5. **Submission**: `submissions/` klasöründeki CSV dosyalarını kullanın

### Kod Modifikasyonu
- Model parametrelerini değiştirmek için: [src/model_builders.py](src/model_builders.py)
- Veri işleme pipeline'ını değiştirmek için: [src/data_utils.py](src/data_utils.py)
- Değerlendirme metriklerini özelleştirmek için: [src/evaluation_utils.py](src/evaluation_utils.py)

### Performans İyileştirme
- Random Forest'ta `n_jobs=-1` tüm CPU çekirdeklerini kullanır
- Daha hızlı eğitim için `n_estimators` sayısını azaltabilirsiniz
- Bellek tasarrufu için `max_depth` değerini düşürün

---

## 📊 Örnek Çıktılar

Modeller çalıştırıldığında aşağıdaki çıktılar üretilir:

### Decision Tree
- **Konsol**: Detaylı metrikler, confusion matrix, feature importance
- **Görseller**: 5 farklı görselleştirme (analiz, ağaç yapısı, metrikler)
- **Submission**: Test seti tahminleri CSV formatında

### Random Forest
- **Konsol**: 100 ağaç istatistikleri, ensemble metrikleri
- **Görseller**: Ağaç örnekleri, istatistikler, performans grafikleri
- **Submission**: Test seti tahminleri CSV formatında

### Model Karşılaştırma
- **Konsol**: Yan yana metrik karşılaştırması, overfitting analizi
- **Görseller**: 7 farklı karşılaştırma grafiği
- **Analiz**: Train vs Validation performans karşılaştırması

---

## �📧 İletişim ve Katkı

Bu proje, Decision Tree ve Random Forest algoritmalarının pratik uygulamasını ve karşılaştırmasını göstermek amacıyla geliştirilmiştir.

**Geliştirme Tarihi:** Aralık 2025

### Katkı Sağlama
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

---

## 📜 Lisans

Bu proje eğitim amaçlıdır ve herkes tarafından kullanılabilir.

---

## 🙏 Teşekkürler

Scikit-learn, Pandas ve diğer açık kaynak kütüphanelerin geliştiricilerine teşekkürler.
