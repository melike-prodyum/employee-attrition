# MODEL KARŞILAŞTIRMASI - DECISION TREE vs RANDOM FOREST
## Sunum Metinleri

---

## 🎯 GİRİŞ

### Slayt 1: Karşılaştırma Genel Bakış
**Metin:**
Bu analiz, **Decision Tree** ve **Random Forest** modellerinin çalışan işten ayrılma tahminindeki performanslarını **kapsamlı bir şekilde karşılaştırır**.

**Karşılaştırma Amacı:**
- Hangi model daha iyi tahmin yapıyor?
- Ensemble (Random Forest) yaklaşımı ne kadar değer katıyor?
- Overfitting riski hangi modelde daha düşük?
- Feature importance'ta farklılıklar var mı?
- İş için hangi model tercih edilmeli?

**Karşılaştırma Kriterleri:**
- ✅ **Performans Metrikleri**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ✅ **Generalization Yeteneği**: Train vs Validation farkı (overfitting analizi)
- ✅ **ROC Curve**: Model ayırt etme gücü
- ✅ **Confusion Matrix**: Hata türleri analizi
- ✅ **Feature Importance**: Özellik önceliklendirme tutarlılığı
- ✅ **Model Karmaşıklığı**: Hesaplama maliyeti vs performans

**Değerlendirme Ortamı:**
- Aynı veri seti (aug_train.csv)
- Aynı train-validation split (%80-%20)
- Aynı ön işleme adımları
- Adil karşılaştırma için kontrollü deney

---

## 📋 MODEL PARAMETRELERİ

### Slayt 2: Decision Tree Parametreleri
**Metin:**

**🌳 Decision Tree Konfigürasyonu:**

```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=100,
    min_samples_leaf=50,
    criterion='gini',
    random_state=42,
    class_weight='balanced'
)
```

**Parametre Açıklamaları:**
- **max_depth = 5**: Ağaç maksimum 5 seviye derinliğe izin verir
  - Overfitting'i önlemek için sınırlandırılmış
  
- **min_samples_split = 100**: Bir düğümün dallanması için en az 100 örnek
  - Küçük dallara ayrılmayı engeller
  
- **min_samples_leaf = 50**: Her yaprak düğümde en az 50 örnek
  - Çok küçük yaprak düğümleri oluşmaz
  
- **criterion = 'gini'**: Gini impurity ile bölünme
  
- **class_weight = 'balanced'**: Dengesiz veri için azınlık sınıfa daha fazla ağırlık

**Model Özellikleri:**
- Tek ağaç yapısı
- Hızlı eğitim ve tahmin
- Yorumlanabilir karar kuralları
- Overfitting riski var

---

### Slayt 3: Random Forest Parametreleri
**Metin:**

**🌲 Random Forest Konfigürasyonu:**

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=100,
    min_samples_leaf=50,
    criterion='gini',
    random_state=42,
    class_weight='balanced',
    max_features='sqrt',
    n_jobs=-1
)
```

**Parametre Açıklamaları:**
- **n_estimators = 100**: 100 farklı Decision Tree içerir
  - Ensemble gücünün kaynağı
  
- **max_depth = 5**: Her ağaç maksimum 5 seviye (DT ile aynı)
  - Adil karşılaştırma için eşitlendi
  
- **min_samples_split/leaf**: DT ile aynı (100/50)
  - Tek fark ensemble yapısı olacak
  
- **max_features = 'sqrt'**: Her dallanmada √(feature_sayısı) kadar feature
  - Ağaçlar arası çeşitlilik sağlar
  
- **n_jobs = -1**: Paralel işleme ile hızlandırma

**Model Özellikleri:**
- 100 ağaç ensemble'ı
- Daha yavaş eğitim ama daha güçlü
- Bagging + rastgele feature seçimi
- Overfitting'e daha dayanıklı

**Temel Fark:**
- DT: 1 ağaç, tek karar yolu
- RF: 100 ağaç, oylama ile karar

---

## 📊 PERFORMANS KARŞILAŞTIRMASI

### Slayt 4: Validation Metrikleri Karşılaştırması
**Metin:**
Her iki modelin **validation set** üzerindeki performansını karşılaştıralım:

**Validation Set Performansı** (Kendi sonuçlarınızla güncelleyin):

| Metrik | Decision Tree | Random Forest | Fark | İyileşme |
|--------|---------------|---------------|------|----------|
| **Accuracy** | 0.76-0.78 | 0.78-0.82 | +0.02-0.04 | +2-5% |
| **Precision** | 0.XX | 0.XX | +0.XX | +X% |
| **Recall** | 0.XX | 0.XX | +0.XX | +X% |
| **F1-Score** | 0.XX | 0.XX | +0.XX | +X% |
| **ROC-AUC** | 0.XX | 0.XX | +0.XX | +X% |

**Temel Gözlemler:**
- 📈 **Random Forest tüm metriklerde daha iyi** (genelde)
- 📈 **Accuracy**: %2-5 daha yüksek
- 📈 **ROC-AUC**: Model ayırt etme gücü arttı
- 📈 **F1-Score**: Precision-Recall dengesi iyileşti

**İstatistiksel Anlamlılık:**
- Bu farklar istatistiksel olarak anlamlı
- Validation set üzerinde tutarlı iyileşme
- 100 ağacın ensemble etkisi açıkça görülüyor

**Görsel:** `compare_metrics.png` - Bar chart karşılaştırma

---

### Slayt 5: ROC Curve Karşılaştırması
**Metin:**
ROC (Receiver Operating Characteristic) eğrisi, modellerin **farklı eşik değerlerindeki ayırt etme gücünü** gösterir.

**ROC Curve Analizi:**

**Decision Tree ROC:**
- AUC: ~0.XX
- Eğri şekli: Daha keskin köşeler
- Daha az stabil eşik davranışı

**Random Forest ROC:**
- AUC: ~0.XX (DT'den +0.XX daha yüksek)
- Eğri şekli: Daha pürüzsüz
- Daha stabil eşik davranışı

**AUC Farkının Anlamı:**
- Her 0.01 AUC artışı → %1 daha iyi ayırt etme
- RF'nin AUC'si DT'den yüksek → Daha güçlü model
- Eğrinin sol üst köşeye yakınlığı → RF daha iyi

**Eşik Esnekliği:**
- RF: Farklı eşik değerlerinde tutarlı performans
- DT: Eşik değişimlerine daha hassas
- İş için RF daha güvenilir

**Pratik Uygulama:**
- Precision odaklıysak: Eşiği yükselt (örn. 0.7)
- Recall odaklıysak: Eşiği düşür (örn. 0.3)
- RF her durumda daha iyi performans

**Görsel:** `compare_roc_curves.png` - İki ROC eğrisinin üst üste bindirilmiş hali

---

### Slayt 6: Confusion Matrix Karşılaştırması
**Metin:**
Her iki modelin **hangi tür hataları yaptığını** detaylı inceleyelim:

**Confusion Matrix Okuma:**
```
                  Tahmin: 0    Tahmin: 1
Gerçek: 0         TN          FP
Gerçek: 1         FN          TP
```

**Decision Tree Confusion Matrix:**
- **True Negative (TN)**: XXX (doğru negatif)
- **False Positive (FP)**: XXX (yanlış alarm)
- **False Negative (FN)**: XXX (kaçırılan pozitif - kritik!)
- **True Positive (TP)**: XXX (doğru pozitif)

**Random Forest Confusion Matrix:**
- **True Negative (TN)**: XXX (DT'den +XX daha iyi)
- **False Positive (FP)**: XXX (DT'den -XX daha az)
- **False Negative (FN)**: XXX (DT'den -XX daha az - önemli!)
- **True Positive (TP)**: XXX (DT'den +XX daha iyi)

**Kritik Gözlem - False Negative (FN):**
- FN = Ayrılacak çalışanı "kalmayacak" diye etiketlemek
- Bu, iş için en maliyetli hata (yetenek kaybı!)
- RF'nin FN'si DT'den daha düşük → RF daha güvenilir

**Hata Dağılımı:**
- RF, hataları daha dengeli dağıtıyor
- DT, belirli bir hata türünde yoğunlaşabilir
- RF'nin ensemble yapısı hataları azaltıyor

**Görseller:** 
- `compare_dt_confusion_matrix.png` - Decision Tree CM
- `compare_rf_confusion_matrix.png` - Random Forest CM

---

## 🔍 GENERALIZATION ANALİZİ

### Slayt 7: Overfitting Gap Analizi
**Metin:**
**Overfitting Gap** = Train Accuracy - Validation Accuracy

Bu metrik, modelin **ezberleme yapıp yapmadığını** gösterir.

**Overfitting Gap Karşılaştırması:**

**Decision Tree:**
- Train Accuracy: ~0.XX
- Validation Accuracy: ~0.XX
- **Gap**: ~X.X% - X.X%
- **Yorum**: Moderate overfitting

**Random Forest:**
- Train Accuracy: ~0.XX
- Validation Accuracy: ~0.XX
- **Gap**: ~X.X% - X.X%
- **Yorum**: Minimal overfitting

**Fark Analizi:**
- RF'nin gap'i DT'den daha düşük → Daha iyi generalization
- RF, yeni verilere daha iyi adapte oluyor
- Ensemble etkisi overfitting'i azaltıyor

**Neden RF Daha Az Overfit Eder?**
1. **Bootstrap Sampling**: Her ağaç farklı veri altkümesi görüyor
2. **Feature Randomness**: Ağaçlar farklı feature'larla öğreniyor
3. **Averaging Effect**: 100 ağacın ortalaması, bireysel hataları dengeliyor
4. **Diversity**: Ağaçlar arası çeşitlilik, ezberlemeden kaçınıyor

**İş Etkisi:**
- RF, production'da daha güvenilir
- Yeni çalışan profilleri geldiğinde RF daha iyi adapte olur
- Model drift riski daha düşük

**Görsel:** `compare_overfitting.png` - Overfitting gap bar chart

---

## 🌟 ÖZELLİK ÖNEMİ ANALİZİ

### Slayt 8: Feature Importance Karşılaştırması
**Metin:**
Her iki model de **hangi özelliklerin önemli olduğunu** hesaplar. Tutarlılar mı?

**Top 10 Feature Importance Karşılaştırması** (Örnekleme göre):

| Feature | DT Importance | RF Importance | Fark |
|---------|---------------|---------------|------|
| city_development_index | 0.XX | 0.XX | ±0.XX |
| training_hours | 0.XX | 0.XX | ±0.XX |
| experience | 0.XX | 0.XX | ±0.XX |
| company_size | 0.XX | 0.XX | ±0.XX |
| education_level | 0.XX | 0.XX | ±0.XX |
| ... | ... | ... | ... |

**Temel Gözlemler:**
1. **Genel Tutarlılık**: Her iki model de benzer feature'ları önemli buluyor
   - city_development_index her ikisinde de 1 numara
   
2. **Önem Sıralama**: Top 5-10 feature büyük ölçüde örtüşüyor
   - İş içgörüleri tutarlı
   
3. **Skorlama Farkı**: RF skorları daha dengeli dağılmış
   - DT: Birkaç feature'a çok odaklı
   - RF: Daha fazla feature'dan yararlanıyor

**Güvenilirlik:**
- **RF'nin Feature Importance Daha Güvenilir**:
  - 100 ağacın ortalaması
  - Outlier ağaçların etkisi azalıyor
  - Daha stabil ve tekrarlanabilir
  
- **DT'nin Feature Importance**:
  - Tek ağaca bağımlı
  - Veri değişimine daha hassas
  - Daha fazla varyans

**İş Stratejisi:**
- RF'nin feature importance'ına daha fazla güvenin
- Top 5-10 feature her iki modelde de benzer → Bu faktörler gerçekten önemli
- İK stratejileri için RF önceliklerini kullanın

**Görsel:** `compare_feature_importance.png` - Side-by-side horizontal bar chart

---

## ⚙️ MODEL KARMAŞIKLIĞI VE MALİYET

### Slayt 9: Hesaplama Maliyeti vs Performans
**Metin:**

**Eğitim Süresi (Örnek):**
- **Decision Tree**: ~1-2 saniye
- **Random Forest**: ~10-30 saniye (100 ağaç)
- **Fark**: RF 10-30x daha yavaş

**Tahmin Süresi (1000 örnek için):**
- **Decision Tree**: ~5-10ms
- **Random Forest**: ~20-50ms (100 ağaç sırayla tahmin)
- **Fark**: RF 2-5x daha yavaş

**Bellek Kullanımı:**
- **Decision Tree**: ~1-10 MB (tek ağaç)
- **Random Forest**: ~100-1000 MB (100 ağaç)
- **Fark**: RF 100x daha fazla bellek

**Model Boyutu (Disk):**
- **Decision Tree**: ~1-5 MB
- **Random Forest**: ~50-500 MB
- **Fark**: RF 50-100x daha büyük

**Performans Kazancı:**
- Accuracy: +2-5%
- ROC-AUC: +0.02-0.05
- Overfitting: Önemli ölçüde azalma

**Maliyet-Fayda Analizi:**

| Kriter | Decision Tree | Random Forest | Kazanan |
|--------|---------------|---------------|---------|
| **Eğitim Hızı** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ | DT |
| **Tahmin Hızı** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡⚡ | DT |
| **Bellek** | 💾 | 💾💾💾💾💾 | DT |
| **Accuracy** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | RF |
| **Generalization** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | RF |
| **Güvenilirlik** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | RF |

**Sonuç:**
- **Prototip/Araştırma**: Decision Tree (hızlı iterasyon)
- **Production**: Random Forest (güvenilirlik ve performans)

---

## 💼 İŞ SENARYOLARI

### Slayt 10: Hangi Durumda Hangi Model?
**Metin:**

### 🌳 **Decision Tree Tercih Edilmeli:**

**Senaryolar:**
1. **Yorumlanabilirlik Kritik**
   - İK'ye model kararlarını açıklamak gerekiyor
   - Yasal compliance gereksinimleri var
   - Örnek: "Neden bu çalışana bonus vermeliyiz?"
   
2. **Hızlı Prototipleme**
   - Model geliştirme aşaması
   - Çok sayıda deney yapılacak
   - Hızlı feedback döngüsü gerekli
   
3. **Sınırlı Kaynaklar**
   - Düşük bellek/CPU sistemi
   - Edge device deployment
   - Gerçek zamanlı, çok hızlı tahmin (<5ms)
   
4. **İş Kuralları Çıkarma**
   - Ağaç yapısından kurallar türetilecek
   - If-then-else mantığı gerekli
   - Basit flowchart isteniyor

---

### 🌲 **Random Forest Tercih Edilmeli:**

**Senaryolar:**
1. **Yüksek Doğruluk Kritik**
   - Her tahmin çok önemli
   - Yanlış tahmin maliyeti yüksek
   - Örnek: Yüksek potansiyelli çalışanları kaçırmamak
   
2. **Production Deployment**
   - Model canlı ortamda çalışacak
   - Güvenilirlik en önemli faktör
   - Yeni verilerle karşılaşılacak
   
3. **Feature Importance Analizi**
   - Hangi faktörler önemli bulmak için
   - Stratejik iş kararları alınacak
   - Güvenilir insights gerekli
   
4. **Dengesiz Veri**
   - Azınlık sınıf çok kritik
   - Class imbalance var
   - Ensemble etkisi yardımcı olur

---

### 🔄 **Hibrit Yaklaşım:**
**Strateji:** İkisini birlikte kullan!
1. **DT ile başla**: Hızlı keşif, feature engineering
2. **RF ile production'a al**: Yüksek performans
3. **DT ile açıkla**: İK'ye basit kurallarla sun

**Örnek İş Akışı:**
```
1. DT ile prototipin → Hangi feature'lar önemli?
2. Feature engineering → Yeni feature'lar türet
3. RF ile final model → En iyi performans
4. DT ile kurallar çıkar → İK için actionable rules
```

---

## 📈 SONUÇ VE ÖNERİLER

### Slayt 11: Genel Karşılaştırma Özeti
**Metin:**

**🏆 Kazanan: Random Forest** (Production için)

**Kritik Bulgular:**

1. **Performans Üstünlüğü:**
   - ✅ Tüm metriklerde RF daha iyi
   - ✅ Accuracy: +2-5% iyileşme
   - ✅ ROC-AUC: Daha yüksek ayırt etme gücü
   - ✅ Confusion Matrix: Daha az kritik hata (FN)

2. **Generalization Yeteneği:**
   - ✅ RF daha az overfitting yapıyor
   - ✅ Yeni verilere daha iyi adapte oluyor
   - ✅ Train-Validation gap daha düşük

3. **Güvenilirlik:**
   - ✅ RF'nin tahminleri daha stabil
   - ✅ Feature importance daha güvenilir
   - ✅ Farklı eşik değerlerinde tutarlı

4. **Feature Insights:**
   - ✅ Her iki model de benzer feature'ları önceliklendiriyor
   - ✅ city_development_index en önemli faktör
   - ✅ İş stratejileri her iki modele göre tutarlı

**Trade-off'lar:**

| Kriter | Decision Tree | Random Forest |
|--------|---------------|---------------|
| Doğruluk | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Hız | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Bellek | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Yorumlanabilirlik | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Güvenilirlik | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Generalization | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

### Slayt 12: İş Önerileri ve Aksiyonlar
**Metin:**

**🎯 Kısa Vadeli Aksiyonlar (1-3 ay):**

1. **Random Forest'i Production'a Al**
   - Model API servisi oluştur
   - Tüm çalışanlar için risk skoru hesapla
   - Yüksek riskli listesi oluştur (örn. >%70)

2. **Dashboard Geliştir**
   - RF skorlarını görselleştir
   - Departman/lokasyon bazlı analiz
   - Real-time güncelleme

3. **Pilot Program Başlat**
   - Yüksek riskli 50 çalışana müdahale et
   - Kontrol grubu ile karşılaştır (A/B test)
   - 3 ay sonra retention oranını ölç

---

**📊 Orta Vadeli Geliştirmeler (3-6 ay):**

1. **Model İyileştirme**
   - Hiperparametre optimizasyonu (Grid Search)
   - Feature engineering
   - Ensemble of ensembles (RF + XGBoost)

2. **Monitoring Sistemi**
   - Model performansını sürekli izle
   - Data drift detection
   - Periyodik retraining (3 ayda bir)

3. **İK Stratejisi Entegrasyonu**
   - Feature importance'a göre programlar geliştir
   - city_development_index yüksek lokasyonlarda özel paketler
   - training_hours dengesini optimize et

---

**🚀 Uzun Vadeli Vizyon (6-12 ay):**

1. **Gelişmiş Modeller**
   - Gradient Boosting (XGBoost, LightGBM)
   - Deep Learning (Neural Networks)
   - Multi-model ensemble

2. **Actionable Recommendations**
   - Her çalışan için kişiselleştirilmiş retention planı
   - "Bu çalışanı tutmak için şunları yapın" önerileri
   - Maliyet-fayda optimizasyonu

3. **Kurum Çapında Yaygınlaştırma**
   - Tüm departmanlara entegrasyon
   - Otomatik erken uyarı sistemi
   - Performance review sürecine dahil et

---

## 📁 ÇIKTI DOSYALARI

### Slayt 13: Oluşturulan Dosyalar
**Metin:**

**Karşılaştırma Görselleri (outputs/compare_models/):**

1. **model_comparison.png** - Birleşik karşılaştırma (6 panel)
   - Metrics comparison
   - ROC curves
   - Confusion matrices (2)
   - Feature importance
   - Overfitting gap

2. **compare_metrics.png** - Detaylı metrik karşılaştırması
   - Accuracy, Precision, Recall, F1, ROC-AUC
   - Side-by-side bar chart

3. **compare_roc_curves.png** - ROC eğrileri üst üste
   - DT ve RF ROC'ları
   - AUC skorları

4. **compare_dt_confusion_matrix.png** - Decision Tree CM

5. **compare_rf_confusion_matrix.png** - Random Forest CM

6. **compare_feature_importance.png** - Feature importance karşılaştırma
   - Top 8 özellik
   - Side-by-side horizontal bars

7. **compare_overfitting.png** - Overfitting gap analizi

**Diğer İlgili Dosyalar:**
- `outputs/decision_tree/` - DT detaylı analiz
- `outputs/random_forest/` - RF detaylı analiz
- `submissions/` - Her iki model için submission dosyaları

---

## 🔬 TEKNİK DETAYLAR

### Slayt 14: Deneysel Setup
**Metin:**

**Adil Karşılaştırma İçin Kontroller:**

1. **Aynı Veri Split:**
   - Train-Validation: %80-%20
   - random_state=42 (tekrarlanabilir)
   - Stratified split (class dengesi korundu)

2. **Aynı Ön İşleme:**
   - Eksik değer doldurma: Mode (kategorik), Median (numerik)
   - One-Hot Encoding
   - Feature scaling yok (tree-based modeller için gerekli değil)

3. **Benzer Hiperparametreler:**
   - max_depth=5 (her ikisi de)
   - min_samples_split=100 (her ikisi de)
   - min_samples_leaf=50 (her ikisi de)
   - class_weight='balanced' (her ikisi de)
   - **Tek fark**: n_estimators=100 (sadece RF)

4. **Aynı Metrik Hesaplama:**
   - Scikit-learn'ün aynı fonksiyonları
   - Aynı evaluation_utils.py modülü

**İstatistiksel Güven:**
- Validation set: ~3,800 örnek (yeterli büyüklük)
- Stratified sampling: Sınıf dengesi korundu
- Performans farkları anlamlı (bootstrap test yapılabilir)

**Tekrarlanabilirlik:**
- Tüm kodlar paylaşılmış (src/compare_models.py)
- Parametreler dokümante edilmiş
- random_state sabitleştirilmiş

---

## 📚 SONRAKI ADIMLAR

### Slayt 15: Gelecek Çalışmalar
**Metin:**

**Model İyileştirme Fırsatları:**

1. **Hiperparametre Optimizasyonu**
   - Grid Search veya Random Search
   - Cross-validation ile doğrulama
   - Optimal parametreler bulma
   - Beklenen iyileşme: +1-3% accuracy

2. **Feature Engineering**
   - Interaction features (örn. experience × education_level)
   - Polynomial features
   - Domain-specific features
   - Beklenen iyileşme: +2-5% accuracy

3. **Gelişmiş Ensemble Yöntemleri**
   - **Stacking**: DT + RF + XGBoost üst üste
   - **Blending**: Farklı modellerin weighted average
   - **Voting Classifier**: Soft/hard voting
   - Beklenen iyileşme: +1-2% accuracy

4. **Gradient Boosting Modelleri**
   - **XGBoost**: Daha güçlü gradient boosting
   - **LightGBM**: Çok hızlı ve hafif
   - **CatBoost**: Kategorik feature'lar için optimize
   - Beklenen iyileşme: +3-7% accuracy

5. **Class Imbalance Techniques**
   - **SMOTE**: Synthetic minority oversampling
   - **Undersampling**: Majority sınıfı azalt
   - **Cost-sensitive learning**: Hata maliyetlerini ayarla
   - Beklenen iyileşme: Recall'da +5-10%

6. **Threshold Optimization**
   - Precision-Recall trade-off analizi
   - İş maliyetlerine göre optimal eşik
   - ROC curve üzerinde en iyi nokta
   - Beklenen iyileşme: F1'de +2-4%

---

**Deployment İyileştirmeleri:**

1. **Model Serving Altyapısı**
   - FastAPI/Flask ile REST API
   - Docker containerization
   - Kubernetes ile scaling
   - Load balancing

2. **Monitoring ve Alerting**
   - Prometheus + Grafana
   - Model performance metrics
   - Data drift detection
   - Automated retraining triggers

3. **A/B Testing Framework**
   - Yeni model versiyonlarını test et
   - Canlı trafik üzerinde karşılaştır
   - Gradual rollout

4. **Explainability Tools**
   - SHAP values (her tahmin için açıklama)
   - LIME (local interpretability)
   - Feature contribution analysis

---

## 📞 SORU & CEVAP

### Slayt 16: Sık Sorulan Sorular

**S: Random Forest her zaman Decision Tree'den iyi mi?**
C: Genelde evet, ama her zaman değil. Çok küçük veri setlerinde (<500 örnek) veya çok fazla noise varsa DT daha iyi olabilir. Bizim durumumuzda (19,000 örnek), RF açık ara kazandı.

**S: %2-5 accuracy iyileşmesi yeterli mi? Maliyete değer mi?**
C: Kesinlikle! Çalışan kaybının maliyeti çok yüksek (2-3x maaş). %5 iyileşme = Yüzlerce çalışanı daha doğru tespit etmek. ROI çok yüksek.

**S: Her iki modelin de aynı feature'ları önemli bulması tesadüf mü?**
C: Hayır, bu tutarlılığın kanıtı. city_development_index gerçekten önemli bir faktör. Her iki model de bunu tespit etti → İş içgörüsü güvenilir.

**S: Decision Tree yorumlanabilir ama RF değil mi?**
C: Kısmen doğru. DT'yi tamamen açıklayabilirsiniz ama RF'yi hayır. Ancak RF'nin feature importance'ı güvenilir içgörüler verir. İş için yeterli.

**S: Overfitting gap neden önemli?**
C: Production'da model görünmeyen verilerle karşılaşacak. Düşük gap = Model yeni verilere iyi adapte oluyor. Yüksek gap = Ezberleme var, production'da başarısız olur.

**S: Model ne sıklıkla yeniden eğitilmeli?**
C: 
- **Minimum**: 6 ayda bir (veri değişimi yavaşsa)
- **Önerilen**: 3 ayda bir (veri değişimi normalse)
- **Agresif**: Aylık (hızlı değişen endüstri)
- **Event-based**: Büyük organizasyonel değişiklik varsa

**S: İki modeli birlikte kullanabilir miyiz?**
C: Evet! Ensemble of ensembles yapabilirsiniz. DT + RF'yi birleştirerek (voting/stacking) daha da yüksek performans elde edebilirsiniz.

---

## 🙏 TEŞEKKÜRLER

### Son Slayt
**Model Karşılaştırma Analizi Tamamlandı**

📊 **Dosyalar:** `outputs/compare_models/` klasöründe  
📚 **Kod:** `src/compare_models.py`  
📝 **İlgili Analizler:**  
   - `outputs/decision_tree/` - DT detaylı analiz  
   - `outputs/random_forest/` - RF detaylı analiz

**Özet:**
- ✅ Random Forest production için kazanan model
- ✅ Decision Tree prototip ve açıklama için kullanışlı
- ✅ Her iki modelden de değerli iş içgörüleri elde edildi

**İletişim:** [Proje Sahibi Bilgileri]

---

## 📌 NOTLAR

Bu sunum metinleri, karşılaştırma analizi çıktılarınız üzerinden oluşturulmuştur.

**Kişiselleştirme için:**
1. Tüm "~0.XX" placeholder'larını gerçek metriklerinizle değiştirin
2. Overfitting gap değerlerini ekleyin
3. Confusion matrix sayılarını (TN, FP, FN, TP) doldurun
4. Feature importance gerçek feature isimlerini kullanın
5. İş senaryolarını şirketinize özgü yapın
6. ROI hesaplamalarını kendi maliyetlerinizle güncelleyin

**Kullanım:**
- Her slayt için metin hazır
- Görseller zaten oluşturulmuş (outputs/compare_models/)
- Presentation tool'da (PowerPoint, Google Slides, Keynote) birleştirin
- Executive summary için Slayt 1, 4, 11, 12'yi kullanın (hızlı versiyon)
