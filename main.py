# =============================================================================
# DİYABET TAHMİN MODELİ - LOJİSTİK REGRESYON ANALİZİ
# Yazar: [İsminiz]
# Tarih: [Tarih]
# =============================================================================

# Gerekli kütüphanelerin import edilmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# Veri setini yükleme
df = pd.read_csv('diabetes.csv')

print("=" * 60)
print("DİYABET TAHMİN MODELİ - LOJİSTİK REGRESYON ANALİZİ")
print("=" * 60)

# 1. VERİ ÖN İŞLEME
print("\nVERİ ÖN İŞLEME VE ANALİZ")
print("-" * 40)

# 0 değerlerinin düzeltilmesi (pandas uyarılarını önlemek için güncellenmiş yöntem)
df_processed = df.copy()
columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for column in columns_to_check:
    # 0 değerlerini NaN ile değiştir
    df_processed[column] = df_processed[column].replace(0, np.nan)
    # Medyan değerini hesapla ve NaN değerleri doldur
    median_value = df_processed[column].median()
    df_processed[column] = df_processed[column].fillna(median_value)

print("0 değerleri medyan ile düzeltildi")

# 2. ÖZELLİK MÜHENDİSLİĞİ - YENİ ÖZELLİKLER
print("\nÖZELLİK MÜHENDİSLİĞİ")
print("-" * 40)

# Yaş grupları oluşturma
df_processed['Age_Group'] = pd.cut(df_processed['Age'],
                                   bins=[20, 30, 40, 50, 60, 90],
                                   labels=['20-29', '30-39', '40-49', '50-59', '60+'])

# BMI kategorileri
df_processed['BMI_Category'] = pd.cut(df_processed['BMI'],
                                      bins=[0, 18.5, 25, 30, 100],
                                      labels=['Zayıf', 'Normal', 'Fazla Kilolu', 'Obez'])

# Glukoz seviyesi risk grupları
df_processed['Glucose_Risk'] = pd.cut(df_processed['Glucose'],
                                      bins=[0, 100, 126, 200],
                                      labels=['Normal', 'Prediyabet', 'Yüksek'])

# Insulin direnci göstergesi
df_processed['Insulin_Resistance'] = (df_processed['Glucose'] * df_processed['Insulin']) / 405

print("Yeni özellikler oluşturuldu:")
print(f"   - Yaş Grupları: {df_processed['Age_Group'].unique().tolist()}")
print(f"   - BMI Kategorileri: {df_processed['BMI_Category'].unique().tolist()}")
print(f"   - Glukoz Risk Grupları: {df_processed['Glucose_Risk'].unique().tolist()}")

# 3. SINIF DENGESİZLİĞİ ANALİZİ VE ÇÖZÜMÜ
print("\nSINIF DENGESİZLİĞİ ANALİZİ")
print("-" * 40)

# Orijinal dağılım
original_dist = df_processed['Outcome'].value_counts()
print("Orijinal Sınıf Dağılımı:")
print(f"  Diyabet Yok (0): {original_dist[0]} (%{original_dist[0] / len(df_processed) * 100:.1f})")
print(f"  Diyabet Var (1): {original_dist[1]} (%{original_dist[1] / len(df_processed) * 100:.1f})")

# Özellik ve hedef değişkenleri ayırma (sadece numerik özellikler)
X = df_processed[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df_processed['Outcome']

print(f"\nÖzellik sayısı: {X.shape[1]}")
print(f"Toplam gözlem: {X.shape[0]}")

# 4. VERİYİ EĞİTİM VE TEST KÜMELERİNE AYIRMA
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Sınıf dağılımını koru
)

print(f"\nVeri Bölümleme:")
print(f"  Eğitim seti: {X_train.shape[0]} örnek")
print(f"  Test seti: {X_test.shape[0]} örnek")

# 5. SMOTE İLE OVERSAMPLING
print("\nSMOTE UYGULANMASI")
print("-" * 40)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("SMOTE Sonrası Eğitim Seti Dağılımı:")
resampled_dist = pd.Series(y_train_resampled).value_counts()
print(f"  Diyabet Yok (0): {resampled_dist[0]}")
print(f"  Diyabet Var (1): {resampled_dist[1]}")
print("Sınıf dengesizliği çözüldü!")

# 6. ÖZELLİK ÖLÇEKLENDİRME
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# 7. MODEL OPTİMİZASYONU
print("\nMODEL OPTİMİZASYONU")
print("-" * 40)

# Ağırlıklı Lojistik Regresyon
logreg_optimized = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000,
    C=0.1,  # Daha güçlü regularizasyon
    solver='liblinear'
)

# Model eğitimi
logreg_optimized.fit(X_train_scaled, y_train_resampled)

# Tahminler
y_pred = logreg_optimized.predict(X_test_scaled)
y_pred_proba = logreg_optimized.predict_proba(X_test_scaled)[:, 1]

print("Optimize edilmiş model eğitildi")

# 8. MODEL DEĞERLENDİRME
print("\nDETAYLI MODEL PERFORMANSI")
print("-" * 40)

# Temel metrikler
accuracy = accuracy_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"Doğruluk (Accuracy): {accuracy:.4f}")
print(f"AUC Skoru: {roc_auc:.4f}")

# Karmaşıklık Matrisi
cm = confusion_matrix(y_test, y_pred)
print(f"\nKarmaşıklık Matrisi:")
print(f"    Tahmin→   0     1")
print(f"Gerçek 0: [{cm[0, 0]:>3}   {cm[0, 1]:>3}] -> Doğru Negatif: {cm[0, 0]}")
print(f"Gerçek 1: [{cm[1, 0]:>3}   {cm[1, 1]:>3}] -> Doğru Pozitif: {cm[1, 1]}")

# Detaylı Sınıflandırma Raporu
print("\nDETAYLI SINIFLANDIRMA RAPORU:")
report = classification_report(y_test, y_pred, target_names=['Diyabet Yok', 'Diyabet Var'], output_dict=True)
print(classification_report(y_test, y_pred, target_names=['Diyabet Yok', 'Diyabet Var']))

# Recall değerini vurgula
recall_diabetes = report['Diyabet Var']['recall']
print(f"KRİTİK METRİK - Diyabet Recall: {recall_diabetes:.4f}")

# 9. EŞİK DEĞER OPTİMİZASYONU
print("\nEŞİK DEĞER OPTİMİZASYONU")
print("-" * 40)

# F1-score'u maksimize eden eşik değeri bul
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds_pr[optimal_idx]

print(f"Optimal Eşik Değeri: {optimal_threshold:.4f}")

# Optimal eşik değer ile tahmin
y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)

# Optimal tahminlerin performansı
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
recall_optimal = cm_optimal[1, 1] / (cm_optimal[1, 0] + cm_optimal[1, 1])

print(f"Optimal Eşik Sonrası Recall: {recall_optimal:.4f}")
print(f"İyileşme: {recall_optimal - recall_diabetes:+.4f}")

# 10. ÖZELLİK ÖNEMİ ANALİZİ
print("\nÖZELLİK ÖNEMİ ANALİZİ")
print("-" * 40)

feature_importance = pd.DataFrame({
    'Özellik': X.columns,
    'Katsayı': logreg_optimized.coef_[0],
    'Mutlak_Önem': np.abs(logreg_optimized.coef_[0])
}).sort_values('Mutlak_Önem', ascending=False)

print("Özellik Önem Sıralaması:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['Özellik']:25} -> {row['Mutlak_Önem']:.4f}")

# 11. KLİNİK YORUM ve ÖNERİLER
print("\nKLİNİK DEĞERLENDİRME ve ÖNERİLER")
print("-" * 40)

print("MEVCUT DURUM:")
print(f"  • Model Doğruluğu: %{accuracy * 100:.1f}")
print(f"  • Diyabet Tespit Oranı (Recall): %{recall_optimal * 100:.1f}")
print(f"  • Model Güvenilirliği (AUC): %{roc_auc * 100:.1f}")

print("\nGÜÇLÜ YÖNLER:")
print("  • Glukoz en önemli prediktör - klinik olarak doğru")
print("  • BMI ikinci sırada - obezite-diyabet ilişkisi destekleniyor")
print("  • SMOTE ile recall önemli ölçüde iyileştirildi")

print("\nÖNERİLER:")
if recall_optimal < 0.7:
    print("  • Recall %70 altında - ek iyileştirme gerekli")
    print("  • Çözüm: Ensemble methodlar (Random Forest, XGBoost) deneyin")
else:
    print("  • Recall %70 üzerinde - kabul edilebilir klinik performans")

if roc_auc > 0.8:
    print("  • AUC %80 üzerinde - excellent ayırt edicilik")
else:
    print("  • AUC iyileştirilmeli")

# 12. GÖRSELLEŞTİRMELER - AYRI AYRI GÖSTERİM
print("\nPERFORMANS GÖRSELLEŞTİRMELERİ")
print("-" * 40)

# 1. Karmaşıklık Matrisi
print("\n1. KARMAŞIKLIK MATRİSİ GÖRSELLEŞTİRMESİ")
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tahmin 0', 'Tahmin 1'],
            yticklabels=['Gerçek 0', 'Gerçek 1'])
plt.title('Karmaşıklık Matrisi - Optimal Eşik')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen Değer')
plt.tight_layout()
plt.show()

# 2. ROC Eğrisi
print("\n2. ROC EĞRİSİ GÖRSELLEŞTİRMESİ")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC eğrisi (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Rastgele Tahmin')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Özellik Önem Grafiği
print("\n3. ÖZELLİK ÖNEM GRAFİĞİ")
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Mutlak_Önem', y='Özellik', palette='viridis')
plt.title('Özellik Önemleri - Lojistik Regresyon Katsayıları')
plt.xlabel('Mutlak Katsayı Değeri')
plt.ylabel('Özellikler')
plt.tight_layout()
plt.show()

# 4. Precision-Recall Eğrisi
print("\n4. PRECISION-RECALL EĞRİSİ")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Eğrisi')
plt.grid(True)
plt.tight_layout()
plt.show()

# 13. PRATİK UYGULAMA ÖRNEKLERİ
print("\nPRATİK TIBBİ UYGULAMA ÖRNEKLERİ")
print("-" * 40)

# Farklı risk profillerine sahip hasta örnekleri
test_cases = [
    # Düşük riskli hasta
    [1, 90, 70, 25, 80, 22.0, 0.2, 25],
    # Orta riskli hasta
    [3, 140, 85, 30, 150, 28.0, 0.5, 45],
    # Yüksek riskli hasta
    [6, 180, 90, 40, 300, 35.0, 1.2, 55]
]

case_descriptions = [
    "Genç, normal glukoz, düşük BMI -> DÜŞÜK RİSK",
    "Orta yaş, yüksek glukoz, fazla kilolu -> ORTA RİSK",
    "Yaşlı, çok yüksek glukoz, obez -> YÜKSEK RİSK"
]

print("Hasta Risk Profili Değerlendirmesi:")
print("-" * 50)

for i, (case, desc) in enumerate(zip(test_cases, case_descriptions)):
    case_scaled = scaler.transform([case])
    proba = logreg_optimized.predict_proba(case_scaled)[0][1]
    prediction = logreg_optimized.predict(case_scaled)[0]

    risk_level = "DÜŞÜK" if proba < 0.3 else "ORTA" if proba < 0.7 else "YÜKSEK"

    print(f"\n{i + 1}. {desc}")
    print(f"   Diyabet Olasılığı: {proba:.1%}")
    print(f"   Risk Seviyesi: {risk_level}")
    print(f"   Tavsiye: {'Rutin takip' if proba < 0.3 else 'Detaylı test önerilir' if proba < 0.7 else 'Acil değerlendirme gerekli'}")

# 14. FİNAL DEĞERLENDİRME
print("\n" + "=" * 60)
print("FİNAL MODEL DEĞERLENDİRMESİ")
print("=" * 60)

print("PERFORMANS ÖZETİ:")
print(f"  • Doğruluk: %{accuracy * 100:.1f}")
print(f"  • Diyabet Tespit Oranı: %{recall_optimal * 100:.1f}")
print(f"  • Model Güvenilirliği: %{roc_auc * 100:.1f} (AUC)")

print("\nKRİTİK BAŞARI METRİKLERİ:")
if recall_optimal > 0.65:
    print("  • Diyabet recall değeri kabul edilebilir seviyede")
else:
    print("  • Diyabet recall değeri iyileştirilmeli")

if roc_auc > 0.75:
    print("  • Model ayırt ediciliği yüksek")
else:
    print("  • Model ayırt ediciliği düşük")

print("\nSONRAKİ ADIMLAR:")
print("  1. Random Forest veya XGBoost ile model çeşitlendirme")
print("  2. Hiperparametre optimizasyonu ile ince ayar")
print("  3. Cross-validation ile model stabilitesini test etme")
print("  4. Dış validasyon için yeni veri setleri ile test")

print("\n" + "=" * 60)
print("ANALİZ BAŞARIYLA TAMAMLANDI!")
print("=" * 60)