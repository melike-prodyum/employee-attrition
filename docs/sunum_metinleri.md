# ÇALIŞAN İŞTEN AYRILMA TAHMİNİ PROJESİ
## Materyal ve Yöntemler - Sunum Metinleri

---

## 📊 MATERYAL (Kullanılan Araçlar ve Teknolojiler)

### Slayt 1: Programlama Dili ve Geliştirme Ortamı
**Metin:**
Bu projede **Python 3.8** programlama dili kullanılmıştır. Python, makine öğrenmesi projeleri için geniş kütüphane desteği ve kolay kullanımı sayesinde tercih edilmiştir. Geliştirme ortamı olarak **Visual Studio Code** editörü ve **Git** sürüm kontrol sistemi kullanılmıştır.

---

### Slayt 2: Veri Seti
**Metin:**
Projede kullanılan veri seti üç dosyadan oluşmaktadır:
- **aug_train.csv**: Eğitim veri seti - Model eğitimi için kullanılan çalışan bilgileri ve işten ayrılma durumları
- **aug_test.csv**: Test veri seti - Tahmin yapılacak çalışan bilgileri
- **sample_submission.csv**: Örnek sonuç dosyası formatı

Veri setinde çalışanların eğitim durumu, deneyim yılı, şirket büyüklüğü, iş değiştirme geçmişi gibi özellikler bulunmaktadır.

---

### Slayt 3: Kullanılan Python Kütüphaneleri
**Metin:**
Proje geliştirmede kullanılan temel Python kütüphaneleri:

**Veri İşleme:**
- **Pandas** (v1.5.0+): Veri okuma, işleme ve analiz
- **NumPy** (v1.23.0+): Sayısal hesaplamalar ve matris işlemleri

**Makine Öğrenmesi:**
- **Scikit-learn** (v1.2.0+): Makine öğrenmesi modelleri, ön işleme ve değerlendirme

**Görselleştirme:**
- **Matplotlib** (v3.6.0+): Grafik ve görselleştirme oluşturma
- **Seaborn** (v0.12.0+): İstatistiksel veri görselleştirme

---

### Slayt 4: Donanım ve Hesaplama Kaynakları
**Metin:**
Proje, standart bir kişisel bilgisayar üzerinde çalışabilecek şekilde optimize edilmiştir. Random Forest modelinde paralel işleme için **n_jobs=-1** parametresi kullanılarak tüm işlemci çekirdeklerinden faydalanılmıştır. Bu sayede model eğitim süresi önemli ölçüde kısaltılmıştır.

---

## 🔬 YÖNTEMLER (Uygulanan Metodolojiler)

### Slayt 5: Veri Ön İşleme - Özellik Hazırlama
**Metin:**
Veri ön işleme aşamasında şu adımlar uygulanmıştır:

1. **Veri Yükleme**: Train, test ve submission dosyaları sisteme yüklendi
2. **Özellik Ayrıştırma**: 
   - Enrollee ID: Çalışan kimlik numarası (modele dahil edilmedi)
   - Target: İşten ayrılma durumu (0: Kalmış, 1: Ayrılmış)
   - Diğer sütunlar: Modele girdi olarak kullanılan özellikler
3. **Sütun Tipi Belirleme**: Kategorik ve numerik sütunlar otomatik olarak ayrıştırıldı

---

### Slayt 6: Eksik Veri İşleme (Missing Value Handling)
**Metin:**
Veri setindeki eksik değerler sistematik olarak doldurulmuştur:

**Kategorik Değişkenler için:**
- Eksik değerler **'Unknown'** (Bilinmiyor) kategorisi ile dolduruldu
- Bu yöntem, eksik verinin de bir bilgi taşıyabileceği varsayımına dayanır

**Numerik Değişkenler için:**
- Eksik değerler **medyan (ortanca)** değeri ile dolduruldu
- Medyan, aykırı değerlerden etkilenmediği için ortalamadan daha robust bir ölçüdür

---

### Slayt 7: Özellik Kodlama - One-Hot Encoding
**Metin:**
Kategorik değişkenler makine öğrenmesi modellerinde kullanılabilmesi için sayısal değerlere dönüştürülmüştür:

**One-Hot Encoding Yöntemi:**
- Her kategorik değişken, kategorilerinin sayısı kadar yeni binary (0-1) sütuna dönüştürüldü
- Örnek: "Eğitim Seviyesi" → "Eğitim_Lise", "Eğitim_Üniversite", "Eğitim_Yükseklisans"
- Bu yöntem, kategoriler arası yanlış sıralama ilişkisini önler
- **Train ve test setleri tutarlı şekilde** kodlanarak model uyumluluğu sağlandı

---

### Slayt 8: Veri Bölünmesi (Train-Validation Split)
**Metin:**
Model eğitimi ve değerlendirmesi için veri seti bölündü:

**Stratejik Bölünme:**
- **Eğitim Seti**: %80 (Model öğrenme aşaması için)
- **Doğrulama Seti**: %20 (Model performans değerlendirmesi için)
- **Stratify=y**: Target değişkeninin dağılımı her iki sette de korundu
- **Random_state=42**: Tekrar edilebilir sonuçlar için sabit rastgelelik

Bu yöntem, modelin görünmeyen verilerdeki performansını gerçekçi şekilde ölçmemizi sağlar.

---

### Slayt 9: Sınıf Dengesizliği Yönetimi (Class Imbalance)
**Metin:**
Veri setinde işten ayrılanların sayısı, kalanlardan daha azdır. Bu dengesizlik şu şekilde ele alınmıştır:

**Class Weight Balancing:**
- Her iki modelde de **class_weight='balanced'** parametresi kullanıldı
- Bu parametre, az sayıda olan sınıfa (işten ayrılanlar) daha fazla önem verir
- Hesaplama: w = n_samples / (n_classes × n_samples_class)
- Böylece model, azınlık sınıfını öğrenmede daha başarılı olur

---

### Slayt 10: Makine Öğrenmesi Modelleri - Decision Tree
**Metin:**
**Decision Tree (Karar Ağacı) Modeli:**

İlk model olarak Decision Tree Classifier kullanılmıştır. Bu model, bir dizi "evet/hayır" soruları ile verileri sınıflandırır.

**Model Parametreleri:**
- **max_depth=5**: Ağaç maksimum 5 seviye derinliğe sahip (aşırı öğrenmeyi önler)
- **min_samples_split=100**: Bir düğümün bölünebilmesi için en az 100 örnek gerekli
- **min_samples_leaf=50**: Her yaprak düğümde en az 50 örnek bulunmalı
- **criterion='gini'**: Gini impurity bölünme kriteri kullanıldı
- **class_weight='balanced'**: Sınıf dengesizliği düzeltmesi uygulandı

---

### Slayt 11: Makine Öğrenmesi Modelleri - Random Forest
**Metin:**
**Random Forest (Rastgele Orman) Modeli:**

İkinci model olarak Random Forest Classifier kullanılmıştır. Bu model, birçok Decision Tree'nin birleşiminden oluşur (ensemble learning).

**Model Parametreleri:**
- **n_estimators=100**: 100 adet karar ağacı kullanıldı
- **max_depth=3**: Her ağaç maksimum 3 seviye derinliğe sahip (daha konservatif)
- **min_samples_split=300**: Bölünme için 300 örnek gerekli
- **min_samples_leaf=150**: Her yaprakta 150 örnek
- **max_features='sqrt'**: Her bölünmede karekök kadar özellik rastgele seçilir
- **n_jobs=-1**: Tüm CPU çekirdekleri kullanılır (hızlı eğitim)

---

### Slayt 12: Model Değerlendirme Metrikleri
**Metin:**
Her iki modelin performansı aşağıdaki metriklerle değerlendirilmiştir:

1. **Accuracy (Doğruluk)**: Doğru tahminlerin toplam tahminlere oranı
2. **Precision (Kesinlik)**: Pozitif tahminlerin ne kadarının gerçekten pozitif olduğu
3. **Recall (Duyarlılık)**: Gerçek pozitiflerin ne kadarının yakalandığı
4. **F1-Score**: Precision ve Recall'un harmonik ortalaması (dengeli metrik)
5. **ROC-AUC**: Model ayırt etme gücünün genel ölçüsü (0.5-1.0 arası)

Bu metrikler hem eğitim hem de doğrulama setinde hesaplanmıştır.

---

### Slayt 13: Confusion Matrix (Karışıklık Matrisi)
**Metin:**
**Confusion Matrix Analizi:**

Model tahminlerinin detaylı analizi için Confusion Matrix oluşturulmuştur:

- **True Negatives (TN)**: Doğru şekilde "kalmış" tahmin edilenler
- **False Positives (FP)**: Yanlışlıkla "ayrılmış" tahmin edilenler (Tip I Hata)
- **False Negatives (FN)**: Yanlışlıkla "kalmış" tahmin edilenler (Tip II Hata)
- **True Positives (TP)**: Doğru şekilde "ayrılmış" tahmin edilenler

Bu matris, modelin hangi tür hataları yaptığını gösterir.

---

### Slayt 14: ROC Eğrisi ve AUC Analizi
**Metin:**
**ROC (Receiver Operating Characteristic) Eğrisi:**

ROC eğrisi, farklı eşik değerlerinde modelin performansını görselleştirir:

- **X ekseni**: False Positive Rate (Yanlış Pozitif Oranı)
- **Y ekseni**: True Positive Rate (Doğru Pozitif Oranı / Recall)
- **AUC (Area Under Curve)**: Eğrinin altında kalan alan (0.5-1.0)
- AUC = 0.5: Rastgele tahmin
- AUC = 1.0: Mükemmel sınıflandırma

Her iki modelin ROC eğrileri karşılaştırmalı olarak çizilmiştir.

---

### Slayt 15: Özellik Önem Analizi (Feature Importance)
**Metin:**
**Feature Importance (Özellik Önemliliği) Analizi:**

Modellerin tahminlerinde hangi özelliklerin daha önemli olduğu analiz edilmiştir:

- Her özelliğin model kararlarına katkı puanı hesaplanmıştır
- En önemli 10-20 özellik görselleştirilmiştir
- Decision Tree tek bir ağaç, Random Forest ise 100 ağacın ortalaması ile önem puanları verir

Bu analiz, işten ayrılmada en etkili faktörleri belirlememizi sağlar.

---

### Slayt 16: Model Karşılaştırma Metodolojisi
**Metin:**
**Decision Tree vs Random Forest Karşılaştırması:**

İki model sistematik olarak karşılaştırılmıştır:

**Karşılaştırma Kriterleri:**
1. **Performans Metrikleri**: Accuracy, Precision, Recall, F1, ROC-AUC
2. **Eğitim ve Doğrulama Performansı**: Overfitting kontrolü
3. **ROC Eğrileri**: Görsel performans karşılaştırması
4. **Özellik Önemlilikleri**: Farklı modellerin farklı özelliklere verdiği önem
5. **Confusion Matrix**: Hata türlerinin karşılaştırılması

Tüm karşılaştırmalar görselleştirilerek kaydedilmiştir.

---

### Slayt 17: Görselleştirme ve Raporlama
**Metin:**
**Sonuçların Görselleştirme Stratejisi:**

Proje boyunca oluşturulan görselleştirmeler:

1. **Performans Grafikleri**: Train vs Validation metriklerinin karşılaştırması
2. **ROC Eğrileri**: Model ayırt etme yeteneğinin görselleştirilmesi
3. **Confusion Matrix Heatmap**: Hata dağılımının ısı haritası
4. **Feature Importance Bar Chart**: En önemli özelliklerin sıralaması
5. **Model Comparison Charts**: İki modelin yan yana performans karşılaştırması

Tüm görseller **outputs/** klasöründe model bazlı dizinlerde kaydedilmiştir.

---

### Slayt 18: Tahmin ve Submission Oluşturma
**Metin:**
**Test Seti Tahminleri ve Sonuç Dosyası:**

Modeller eğitildikten sonra test veri setinde tahminler yapılmıştır:

1. **Tam Veri ile Yeniden Eğitim**: Model tüm train verisi ile yeniden eğitildi
2. **Test Tahminleri**: Test setindeki her çalışan için işten ayrılma olasılığı hesaplandı
3. **Submission Dosyası**: 
   - Format: enrollee_id, target (olasılık değeri)
   - Decision Tree için: **submission_decision_tree.csv**
   - Random Forest için: **submission_random_forest.csv**
4. Dosyalar **submissions/** klasörüne kaydedildi

---

### Slayt 19: Modüler Kod Yapısı
**Metin:**
**Proje Kod Organizasyonu:**

Proje, sürdürülebilir ve yeniden kullanılabilir bir yapıda geliştirilmiştir:

**Modüler Dosya Yapısı:**
- **data_utils.py**: Veri yükleme, ön işleme, encoding fonksiyonları
- **evaluation_utils.py**: Metrik hesaplama, değerlendirme fonksiyonları
- **model_builders.py**: Model oluşturma ve parametre yönetimi
- **decision_tree_model.py**: Decision Tree ana script
- **random_forest_model.py**: Random Forest ana script
- **compare_models.py**: Model karşılaştırma script

Bu yapı, kod tekrarını önler ve bakımı kolaylaştırır.

---

### Slayt 20: Sonuç ve Bulgular
**Metin:**
**Proje Sonuçları:**

Bu projede çalışan işten ayrılma tahmini için iki farklı makine öğrenmesi modeli geliştirilmiş ve karşılaştırılmıştır:

**Temel Bulgular:**
- Decision Tree ve Random Forest modelleri başarıyla uygulanmıştır
- Sınıf dengesizliği sorunu class_weight ile çözülmüştür
- Modeller detaylı metriklerle değerlendirilmiştir
- En önemli çalışan özellikleri belirlenmiştir
- Her iki model için test tahminleri üretilmiştir

**Proje Çıktıları:**
- Detaylı görselleştirmeler ve analizler
- İki farklı submission dosyası
- Yeniden kullanılabilir modüler kod yapısı

---

## 📌 ÖZET TABLO: MATERYAL VE YÖNTEMLER

### Materyal Özeti:
| Kategori | Kullanılan Araç/Teknoloji |
|----------|---------------------------|
| Programlama Dili | Python 3.8+ |
| Veri İşleme | Pandas, NumPy |
| Makine Öğrenmesi | Scikit-learn |
| Görselleştirme | Matplotlib, Seaborn |
| Geliştirme Ortamı | VS Code, Git |
| Veri Seti | aug_train.csv, aug_test.csv |

### Yöntemler Özeti:
| Aşama | Uygulanan Yöntem |
|-------|------------------|
| Veri Ön İşleme | Eksik değer doldurma (medyan/unknown), One-Hot Encoding |
| Veri Bölünmesi | Train-Validation Split (80-20, stratified) |
| Dengesizlik Yönetimi | Class Weight Balancing |
| Modelleme | Decision Tree, Random Forest (ensemble) |
| Değerlendirme | Accuracy, Precision, Recall, F1, ROC-AUC |
| Analiz | Confusion Matrix, ROC Curve, Feature Importance |
| Karşılaştırma | Çok metrikli model karşılaştırması |
| Görselleştirme | Grafik, heatmap, bar chart |

---

## 🎯 SUNUM İPUÇLARI

### Her Slayt İçin Öneriler:
1. **Başlıkları vurgulayın**: Kalın puntolu başlıklar kullanın
2. **Madde işaretleri**: Ana metni madde işaretlerine dönüştürün
3. **Görseller ekleyin**: outputs/ klasöründeki grafikleri kullanın
4. **Örneklerle destekleyin**: Veri setinden örnekler gösterin
5. **Akış sağlayın**: Materyal → Yöntemler → Sonuçlar sıralamasını koruyun

### Zaman Dağılımı Önerisi (20 slayt için):
- Materyal slaytları: 3-4 dakika
- Yöntem slaytları: 10-12 dakika
- Sonuç ve özet: 3-4 dakika
- Sorular: 2-3 dakika

**Toplam Sunum Süresi: 18-23 dakika**

---

**Not:** Bu doküman, projenizin materyal ve yöntemler bölümü için hazırlanmış detaylı sunum metinlerini içermektedir. Her slayt metni doğrudan kullanılabilir veya kendi sunum tarzınıza göre uyarlayabilirsiniz.
