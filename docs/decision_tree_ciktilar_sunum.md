# DECISION TREE MODELİ - ÇIKTILAR VE SONUÇLAR
## Sunum Metinleri

---

## 🎯 GİRİŞ

### Slayt 1: Decision Tree Modeli Genel Bakış
**Metin:**
Decision Tree (Karar Ağacı) modelimiz, çalışan işten ayrılma tahmininde ilk temel modelimizdir. Bu model, bir dizi **"evet/hayır" soruları** ile verileri sınıflandırır ve **yorumlanabilirliği** sayesinde iş dünyası için değerli içgörüler sağlar.

**Model Özellikleri:**
- **Algoritma**: Decision Tree Classifier (Scikit-learn)
- **Amaç**: Çalışanların işten ayrılma olasılığını tahmin etmek
- **Avantajı**: Basit, görsel olarak anlaşılabilir, iş kuralları çıkarılabilir
- **Üretilen Çıktılar**: 8 farklı görsel analiz + 1 submission dosyası

---

## 📊 MODEL PARAMETRELERİ VE YAPISI

### Slayt 2: Model Parametreleri
**Metin:**
Decision Tree modelimiz, **aşırı öğrenmeyi (overfitting) önlemek** ve **genelleme yeteneğini artırmak** için optimize edilmiş parametrelerle oluşturulmuştur:

**Temel Parametreler:**
- **max_depth = 5**: Ağaç maksimum 5 seviye derine inebildi
  - Fazla derin ağaçlar ezberleme yapar, sığ ağaçlar basit kalmayı sağlar
  
- **min_samples_split = 100**: Bir düğümün dallanabilmesi için en az 100 örnek gerekli
  - Küçük dallara ayrılmayı engeller
  
- **min_samples_leaf = 50**: Her yaprak düğümde en az 50 örnek olmalı
  - Çok küçük yapraklara sahip olmayı önler
  
- **criterion = 'gini'**: Gini impurity bölünme kriteri
  - Her bölünmede en iyi ayrımı yapacak özellik seçilir

**Dengesizlik Yönetimi:**
- **class_weight = 'balanced'**: Azınlık sınıfa (işten ayrılanlar) daha fazla ağırlık verildi

---

### Slayt 3: Ağaç Yapısı İstatistikleri
**Metin:**
Eğitim sonrası modelimizin yapısal özellikleri:

**Ağaç İstatistikleri:**
- **Gerçek Derinlik**: Model eğitim sonrası gerçek derinliğe ulaştı
- **Yaprak Sayısı**: Modelin son karar noktalarının sayısı
- **Toplam Düğüm Sayısı**: Ağaçtaki tüm karar noktaları ve yapraklar

Bu sayılar, modelimizin **karmaşıklık seviyesini** gösterir:
- Çok fazla yaprak = Aşırı öğrenme riski
- Çok az yaprak = Yetersiz öğrenme (underfitting)
- Bizim modelimiz = Dengeli yapı

**Görsel:** `dt_tree_structure_simple.png` - İlk 3 seviye karar yapısı

---

## 📈 PERFORMANS METRİKLERİ

### Slayt 4: Model Performans Skorları
**Metin:**
Decision Tree modelimiz hem eğitim hem de doğrulama setlerinde değerlendirilmiştir:

**Train Set Performansı:**
- **Accuracy (Doğruluk)**: Doğru tahminlerin yüzdesi
- **Precision (Kesinlik)**: "Ayrılacak" dediğimizde ne kadar isabetliyiz
- **Recall (Duyarlılık)**: Gerçekten ayrılanların ne kadarını yakaladık
- **F1-Score**: Precision ve Recall'un dengeli ölçüsü

**Validation Set Performansı:**
- Modelin **görünmeyen veriler**deki gerçek performansı
- Train'den daha düşük olması normaldir (generalization)
- İki set arasında çok büyük fark olmaması önemli (overfitting kontrolü)

**Görsel:** `dt_performance_metrics.png` - Train vs Validation karşılaştırması

---

### Slayt 5: Performans Değerlendirmesi - Sayısal Sonuçlar
**Metin:**
**Validation Set Sonuçları (Modelin Gerçek Performansı):**

Örnek değerler (kendi çalıştırdığınız sonuçlara göre güncelleyin):
- **Accuracy**: ~0.76-0.78 (Tüm tahminlerin %76-78'i doğru)
- **Precision**: ~0.XX (Pozitif tahminlerimizin isabeti)
- **Recall**: ~0.XX (İşten ayrılanları yakalama oranımız)
- **F1-Score**: ~0.XX (Dengeli performans)
- **ROC-AUC**: ~0.XX (0.5'in üzerinde = Rastgele tahmin'den iyi)

**Yorum:**
- Model, çalışanların çoğunu doğru sınıflandırıyor
- Dengesiz veri seti nedeniyle Precision ve Recall arasındaki trade-off kritik
- Class balancing sayesinde azınlık sınıf (işten ayrılanlar) da öğrenildi

---

### Slayt 6: ROC Eğrisi Analizi
**Metin:**
**ROC (Receiver Operating Characteristic) Eğrisi:**

ROC eğrisi, farklı eşik değerlerinde modelin **ayırt etme gücünü** gösterir.

**Eğri Özellikleri:**
- **X Ekseni**: False Positive Rate (Yanlış alarm oranı)
- **Y Ekseni**: True Positive Rate (Doğru tespit oranı / Recall)
- **Köşegen Çizgi**: Rastgele tahmin çizgisi (AUC=0.50)
- **Kırmızı Eğri**: Modelimizin performans eğrisi

**AUC (Area Under Curve) Yorumu:**
- **AUC = 0.50**: Model rastgele tahmin ediyor
- **AUC = 1.00**: Mükemmel ayrım yapıyor
- **Bizim AUC**: ~0.XX (kendi sonucunuza göre)

**Anlam:** Eğri sol üst köşeye ne kadar yakınsa, model o kadar başarılı demektir.

**Görsel:** `dt_roc_curve.png` - ROC eğrisi ve AUC skoru

---

## 🔍 DETAYLI ANALİZ ÇIKTILARI

### Slayt 7: Confusion Matrix (Karışıklık Matrisi)
**Metin:**
**Confusion Matrix - Modelin Hatalarını Anlamak:**

Confusion Matrix, modelimizin **hangi tür hataları yaptığını** detaylı gösterir:

```
                   Tahmin
              Not Leave  |  Leave
        ──────────────────┼─────────
Gerçek  Not Leave   TN   |   FP
        Leave       FN   |   TP
```

**Dört Tür Sonuç:**
1. **True Negatives (TN)**: Kalmış + Kalmış tahmin ✓ (Doğru)
2. **False Positives (FP)**: Kalmış + Ayrılacak tahmin ✗ (Yanlış Alarm)
3. **False Negatives (FN)**: Ayrılmış + Kalacak tahmin ✗ (Kaçırılmış)
4. **True Positives (TP)**: Ayrılmış + Ayrılacak tahmin ✓ (Doğru)

**İş Açısından:**
- **FP (Yanlış Alarm)**: Kalacak çalışana gereksiz müdahale
- **FN (Kaçırma)**: Ayrılacak çalışanı fark edememe (daha kritik!)

**Görsel:** `dt_confusion_matrix.png` - Isı haritası ile görselleştirilmiş matris

---

### Slayt 8: Detaylı Sınıflandırma Raporu
**Metin:**
**Classification Report - Sınıf Bazlı Performans:**

Model, her iki sınıf için ayrı ayrı değerlendirildi:

**Class 0 (Not Leave - Kalmışlar):**
- Precision: İddia ettiğimiz "kalmışlar"ın gerçekten kalma oranı
- Recall: Gerçekten kalanların ne kadarını bulduk
- F1-Score: İkisinin dengesi
- Support: Bu sınıftan kaç örnek var

**Class 1 (Leave - Ayrılmışlar):**
- Precision: İddia ettiğimiz "ayrılanlar"ın gerçekten ayrılma oranı
- Recall: Gerçekten ayrılanların ne kadarını yakaladık
- F1-Score: İkisinin dengesi
- Support: Bu sınıftan kaç örnek var

**Weighted Average:**
- Sınıf büyüklüklerine göre ağırlıklı ortalama metrikler
- Genel model performansının dengeli göstergesi

---

## 🏆 ÖZELLİK ÖNEMLİLİĞİ (FEATURE IMPORTANCE)

### Slayt 9: En Önemli Özellikler - Top 10
**Metin:**
**Feature Importance - İşten Ayrılmada En Etkili Faktörler:**

Decision Tree modeli, tahminlerinde **hangi özellikleri en çok kullandığını** bize söyler. Bu, iş açısından çok değerli içgörüler sağlar!

**Önem Puanı Nasıl Hesaplanır?**
- Her özellik, ağaçtaki bölünmelerde ne kadar "bilgi kazancı" sağladıysa o kadar önemlidir
- Toplamları 1.0'a eşittir
- Yüksek puan = O özellik çok sık ve etkili kullanıldı

**Top 10 Özellik Yorumu:**
İlk sıralardaki özellikler işten ayrılmada **en belirleyici faktörler**dir:
- Bu özelliklere yönelik insan kaynakları politikaları geliştirilebilir
- Risk altındaki çalışanları erken tespit etmek için bu faktörler izlenebilir
- Şirket stratejileri bu içgörülere göre şekillendirilebilir

**Görsel:** `dt_feature_importance.png` - Bar chart ile top 10 özellik

---

### Slayt 10: Feature Importance'ın İş Değeri
**Metin:**
**Özellik Önemliliği Analizinin İş Uygulamaları:**

Feature Importance çıktılarından şu aksiyonlar alınabilir:

**1. Yüksek Önem → Öncelikli Müdahale:**
   - En önemli faktörlerde iyileştirme yapılırsa, işten ayrılma azalır
   - Örnek: Eğer "deneyim_yılı" çok önemliyse → Kariyer gelişim programları

**2. Düşük Önem → Kaynak Tasarrufu:**
   - Az önemli faktörlere gereksiz kaynak harcanmaz
   - Veri toplama süreçleri sadeleştirilebilir

**3. Sürpriz Faktörler:**
   - Beklenmedik şekilde önemli çıkan özellikler → Yeni içgörüler
   - Bu faktörlerin neden önemli olduğu araştırılmalı

**4. Model Sadeleştirme:**
   - Çok az önemli özellikleri çıkararak model basitleştirilebilir
   - Performans kaybı olmadan daha hızlı tahminler yapılabilir

---

## 🌳 AĞAÇ YAPISI GÖRSELLEŞTİRMELERİ

### Slayt 11: Karar Ağacı Görselleştirmesi - İlk 3 Seviye
**Metin:**
**Decision Tree Yapısı - Karar Akışını Görmek:**

Decision Tree'nin en büyük avantajı: **Görsel olarak yorumlanabilir olması!**

**Basitleştirilmiş Görselleştirme (İlk 3 Seviye):**
- Her **dikdörtgen kutu** = Bir karar düğümü
- **Üstteki soru** = Hangi özelliğe hangi değerde bölünüyor
- **Renkler**: Turuncu = Leave eğilimli, Mavi = Not Leave eğilimli
- **Samples**: O düğümde kaç örnek var
- **Value**: [Not Leave sayısı, Leave sayısı]
- **Gini**: Düğümün saflık ölçüsü (0 = tamamen saf, 0.5 = karışık)

**Okuma Örneği:**
"Eğer özellik_X ≤ 0.5 ise → Sol dala git, değilse → Sağ dala git"

**Görsel:** `dt_tree_structure_simple.png` - İlk 3 seviye karar akışı

---

### Slayt 12: Tam Ağaç Yapısı
**Metin:**
**Tam Decision Tree Görselleştirmesi:**

Modelimizin **tüm karar yapısı** detaylı görselleştirilmiştir.

**Tam Ağaç Görseli Özellikleri:**
- **5 seviye derinlikte** tüm dal ve yapraklar gösterilir
- Her yaprak (en alt düğümler) = Bir **son karar noktası**
- Ağacın sol tarafı genelde bir tür, sağ tarafı diğer tür eğilimlidir

**Kullanım Alanları:**
1. **İş Kuralları Çıkarmak**: 
   - Bir yaprağa giden yolu takip edin → Bu bir iş kuralıdır!
   - Örnek: "Eğer deneyim < 2 ve eğitim = lisans ve ... ise → Ayrılma riski %80"

2. **Manuel Doğrulama**:
   - Bazı kuralların mantıklı olup olmadığını kontrol edebilirsiniz
   - Domain bilgisiyle tutarsız kurallar varsa model revize edilebilir

3. **Sunum İçin**:
   - Yöneticilere "model böyle düşünüyor" diye gösterilebilir

**Görsel:** `decision_tree_full.png` - 5 seviyeli tam ağaç yapısı

---

## 📊 DİĞER GÖRSEL ÇIKTILAR

### Slayt 13: Target Dağılımı
**Metin:**
**Veri Setindeki Sınıf Dağılımı:**

Modelimizin üzerinde çalıştığı veri setinin hedef değişken dağılımı:

**Target Distribution (Train Set):**
- **Class 0 (Not Leave)**: Kaç çalışan işte kalmış
- **Class 1 (Leave)**: Kaç çalışan işten ayrılmış

**Dengesizlik Durumu:**
- Eğer iki sınıf arasında büyük fark varsa → **Imbalanced Dataset**
- Bu projede **class_weight='balanced'** ile dengelendi
- Dengeleme yapılmasaydı model sadece çoğunluk sınıfını öğrenirdi

**Görsel Yorumu:**
- Bar grafikteki yükseklik farkı ne kadar fazlaysa dengesizlik o kadar büyük
- Modelimiz bu dengesizliği dikkate alarak eğitildi

**Görsel:** `dt_target_distribution.png` - Sınıf dağılım bar grafiği

---

### Slayt 14: Birleşik Analiz Görseli
**Metin:**
**Tüm Analizlerin Tek Görselde Özeti:**

Tüm temel analizler **tek bir görselde** birleştirilmiştir - sunum ve rapor için ideal!

**Birleşik Görseldeki 6 Grafik:**

1. **Confusion Matrix** (Sol üst): Tahmin doğruluğu detayı
2. **Feature Importance** (Orta üst): En önemli 10 özellik
3. **ROC Curve** (Sağ üst): Ayırt etme gücü
4. **Target Distribution** (Sol alt): Veri dengesi
5. **Performance Metrics** (Orta alt): Train vs Validation
6. **Tree Structure** (Sağ alt): Ağaç yapısı (3 seviye)

**Kullanım:**
- Rapor ekinde tek sayfa özet olarak kullanılabilir
- Sunumda "genel bakış" slaytı olarak gösterilebilir
- Yöneticilere hızlı bilgilendirme için uygun

**Görsel:** `decision_tree_analysis.png` - 6'lı birleşik analiz

---

## 🎯 TEST TAHMİNLERİ VE SUBMISSION

### Slayt 15: Test Seti Tahminleri
**Metin:**
**Görünmeyen Veriler Üzerinde Tahmin:**

Model eğitildikten sonra, **test veri setinde** tahminler yapılmıştır:

**Test Tahmin Süreci:**
1. **Final Model Eğitimi**: 
   - Model tüm train verisi (train + validation birleşik) ile yeniden eğitildi
   - Maksimum veri kullanarak en iyi öğrenme sağlandı

2. **Test Tahminleri**:
   - Test setindeki her çalışan için işten ayrılma **olasılığı** hesaplandı
   - Çıktı: 0.0 - 1.0 arası olasılık değerleri
   - 0.0 = Kesinlikle kalmayacak, 1.0 = Kesinlikle ayrılacak

3. **Tahmin İstatistikleri**:
   - Ortalama olasılık: Test setinin genel risk seviyesi
   - Standart sapma: Tahminlerdeki çeşitlilik
   - Min/Max: En düşük ve en yüksek risk skorları

---

### Slayt 16: Submission Dosyası
**Metin:**
**Tahmin Sonuçlarının Kaydedilmesi:**

Test tahminleri **submission formatında** kaydedilmiştir:

**Dosya Yapısı:** `submission_decision_tree.csv`
```
enrollee_id,target
1234,0.1234
5678,0.8765
...
```

**Sütunlar:**
- **enrollee_id**: Çalışan kimlik numarası
- **target**: İşten ayrılma olasılığı (0.0-1.0)

**Kullanım Alanları:**
1. **Kaggle/Yarışma**: Varsa kaggle yarışmasına submit edilebilir
2. **İş Uygulaması**: Risk skorlarına göre çalışanlar önceliklendirilebilir
   - Yüksek skor (>0.7) = Yüksek risk → Acil müdahale
   - Orta skor (0.3-0.7) = Orta risk → İzleme
   - Düşük skor (<0.3) = Düşük risk → Rutin takip
3. **Dashboard**: Skorlar görselleştirilerek dashboard'a aktarılabilir

**Dosya Konumu:** `submissions/submission_decision_tree.csv`

---

## 📁 ÇIKTI DOSYALARI ÖZETİ

### Slayt 17: Üretilen Tüm Dosyalar
**Metin:**
**Decision Tree Model Çıktıları - Dosya Envanteri:**

**1. Birleşik Görsel (1 dosya):**
- ✅ `decision_tree_analysis.png` - 6'lı kombinasyon grafik

**2. Tekil Görseller (7 dosya):**
- ✅ `dt_confusion_matrix.png` - Karışıklık matrisi
- ✅ `dt_feature_importance.png` - Özellik önemliliği
- ✅ `dt_roc_curve.png` - ROC eğrisi
- ✅ `dt_target_distribution.png` - Hedef dağılımı
- ✅ `dt_performance_metrics.png` - Performans karşılaştırması
- ✅ `dt_tree_structure_simple.png` - Basit ağaç (3 seviye)
- ✅ `decision_tree_full.png` - Tam ağaç yapısı (5 seviye)

**3. Tahmin Dosyası (1 dosya):**
- ✅ `submission_decision_tree.csv` - Test tahminleri

**Toplam: 9 dosya** (8 görsel + 1 CSV)

**Dosya Konumları:**
- Görseller: `outputs/decision_tree/`
- Submission: `submissions/`

---

## 💡 MODEL DEĞERLENDİRME VE YORUMLAR

### Slayt 18: Decision Tree'nin Avantajları
**Metin:**
**Decision Tree Modelinin Güçlü Yanları:**

✅ **1. Yorumlanabilirlik:**
   - Ağaç görselleştirmesi ile karar süreci görülebilir
   - İş kuralları kolayca çıkarılabilir
   - Teknik olmayan kişilere açıklanabilir

✅ **2. Veri Ön İşleme Toleransı:**
   - Feature scaling gerektirmez
   - Kategorik ve numerik verilerle doğrudan çalışabilir
   - Outlier'lara (aykırı değerlere) görece robust

✅ **3. Hızlı Eğitim:**
   - Küçük-orta ölçekli verilerde çok hızlı eğitilir
   - Real-time güncellemeler için uygun

✅ **4. Feature Importance:**
   - Hangi özelliklerin önemli olduğunu doğrudan verir
   - Veri biliminin iş değerine dönüşmesini kolaylaştırır

✅ **5. Non-linear İlişkiler:**
   - Doğrusal olmayan karmaşık ilişkileri yakalayabilir
   - Özellik etkileşimlerini otomatik öğrenir

---

### Slayt 19: Decision Tree'nin Dezavantajları
**Metin:**
**Decision Tree Modelinin Zayıf Yanları:**

❌ **1. Overfitting Riski:**
   - Çok derin ağaçlar veriye ezberler, genelleştiremez
   - Bu projede max_depth=5 ile sınırlandırıldı
   - Pruning (budama) teknikleri uygulandı

❌ **2. Instability (Kararsızlık):**
   - Veri setinde küçük değişiklikler ağaç yapısını çok değiştirebilir
   - Farklı random_state'lerde farklı ağaçlar oluşabilir
   - Random Forest ile bu sorun giderilir

❌ **3. Bias to Dominant Classes:**
   - Dengesiz verilerde çoğunluk sınıfına yönelir
   - class_weight='balanced' ile düzeltildi

❌ **4. Yerel Optimal:**
   - Her bölünmede yerel en iyi seçim yapılır (greedy)
   - Global optimum garanti edilmez

**Çözüm:** Bu dezavantajlar Random Forest modelinde büyük ölçüde giderilir!

---

### Slayt 20: Sonuç ve Öneriler
**Metin:**
**Decision Tree Modeli - Sonuç ve Öneriler:**

**🎯 Model Başarısı:**
- Model, çalışan işten ayrılma tahmininde **anlamlı sonuçlar** üretti
- Validation set performansı **kabul edilebilir seviyede**
- Özellik önemliliği analizi **değerli içgörüler** sağladı
- Tüm çıktılar **görselleştirildi ve kaydedildi**

**📊 Öne Çıkan Bulgular:**
- En önemli faktörler belirlendi (feature importance)
- Dengesiz veri sorunu başarıyla yönetildi
- Model yorumlanabilir ve açıklanabilir

**🔮 Sonraki Adımlar:**
1. **Random Forest ile Karşılaştırma**: Ensemble learning ile performans artışı
2. **Hiperparametre Optimizasyonu**: Grid search ile daha iyi parametreler
3. **İş Entegrasyonu**: Risk skorlarının HR sistemine entegrasyonu
4. **Sürekli İzleme**: Yeni verilerle modelin güncellenmesi

**💼 İş Önerisi:**
Çıkan risk skorlarına göre çalışan elde tutma (retention) stratejileri geliştirilmeli!

---

## 📌 HIZLI REFERANS: DOSYA - SLAYT EŞLEŞTİRMESİ

### Hangi Görseli Hangi Slayta Eklemeliyim?

| Slayt No | Slayt Konusu | Eklenecek Görsel Dosya |
|----------|--------------|------------------------|
| 3 | Ağaç Yapısı İstatistikleri | `dt_tree_structure_simple.png` |
| 4 | Model Performans Skorları | `dt_performance_metrics.png` |
| 6 | ROC Eğrisi Analizi | `dt_roc_curve.png` |
| 7 | Confusion Matrix | `dt_confusion_matrix.png` |
| 9 | En Önemli Özellikler | `dt_feature_importance.png` |
| 11 | Karar Ağacı (İlk 3 Seviye) | `dt_tree_structure_simple.png` |
| 12 | Tam Ağaç Yapısı | `decision_tree_full.png` |
| 13 | Target Dağılımı | `dt_target_distribution.png` |
| 14 | Birleşik Analiz | `decision_tree_analysis.png` |

**Tüm görseller:** `outputs/decision_tree/` klasöründe

---

## 🎓 SUNUM İPUÇLARI

### Etkili Sunum İçin Öneriler:

**1. Görsel Kullanımı:**
- Her slayta **tek bir odak grafik** ekleyin
- Grafiği açıklarken **ok işaretleri** kullanarak önemli noktaları vurgulayın
- Renkli yazıcıda basılırsa etkisi artar

**2. Zaman Yönetimi:**
- Giriş (Slayt 1-3): 2-3 dakika
- Performans (Slayt 4-8): 4-5 dakika
- Özellik Analizi (Slayt 9-10): 2-3 dakika
- Ağaç Görselleri (Slayt 11-12): 2-3 dakika
- Diğer Çıktılar (Slayt 13-17): 3-4 dakika
- Değerlendirme (Slayt 18-20): 2-3 dakika

**3. Hikaye Akışı:**
"Model Oluşturduk → Eğittik → Değerlendirdik → Sonuçları Analiz Ettik → İş Değeri Çıkardık"

**4. Teknik Seviye Ayarı:**
- Teknik dinleyiciler için: Tüm detayları anlatın
- İş odaklı dinleyiciler için: Slayt 9-10, 18-20'ye odaklanın
- Karma grup için: Basit başlayıp isteğe göre derinleştirin

**5. Soru Cevap Hazırlığı:**
- "Neden Decision Tree?" → Yorumlanabilirlik
- "Overfitting var mı?" → max_depth=5 ile önlendi
- "İş değeri nedir?" → Feature importance'tan çıkan içgörüler

---

## 📝 OPSIYONEL: DEMO SENARYOSU

### Canlı Demo Yapmak İsterseniz:

**Senaryo: "Bir Çalışan Üzerinde Model Testi"**

1. Test setinden bir çalışan profili seçin
2. Özelliklerini ekrana yansıtın (anonim)
3. Modelin tahminini gösterin (ör: Risk = 0.78)
4. `dt_tree_structure_simple.png` üzerinde bu çalışanın hangi yolu izlediğini gösterin
5. "İşte bu yüzden model yüksek risk diyor" deyin
6. İzleyiciler modelin mantığını somut görsün!

---

**Not:** Bu doküman, Decision Tree model çıktılarınızı sunum olarak anlatmanız için hazırlanmıştır. Tüm metinler doğrudan kullanılabilir veya kendi tarzınıza göre uyarlanabilir. Görselleri PowerPoint/Google Slides'a ekleyerek profesyonel bir sunum hazırlayabilirsiniz.
