# Müşteri Segmentasyonu (KMeans)

Müşterileri gelir ve harcama davranışına göre segmentlere ayırmak için KMeans kümeleme algoritmasını kullanan bir model.

- Veri seti: Sentetik (numpy ile üretildi)
- Algoritma: KMeans Clustering

## Nasıl Çalışır

- `age`, `annual_income_k` ve `spending_score` içeren 300 örneklik sentetik bir veri seti oluşturulur.
- `annual_income_k` ve `spending_score` özellikleri `StandardScaler` ile ölçeklenir.
- k=2..8 aralığında her küme sayısı için silhouette skoru hesaplanır ve en iyi k seçilir.
- Seçilen k ile KMeans modeli eğitilir, küme boyutları ve merkezleri yazdırılır.
- Örnek bir müşteri (gelir=80k, harcama skoru=75) için tahmini küme gösterilir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python customer_segmentation.py
```
