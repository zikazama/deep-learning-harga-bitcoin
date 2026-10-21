# Multivariate Multi-Horizon Time Series Forecasting Harga Bitcoin

Proyek ini memprediksi harga penutupan (`Close`) Bitcoin **24 jam ke depan sekaligus** (multi-step) dari beberapa fitur sekaligus (multivariate). Model dilatih dengan *custom training loop* (`tf.GradientTape`).

## Dataset

*Multivariate Crypto Data Hourly since 2017 to 2023*, subset Bitcoin (data per jam).

- Fitur dasar: `Close`, `Volume_USDT`, `RSI`, `MACD_Hist`, `ATR`, `KAMAO`
- Fitur tambahan (rolling statistic 24 jam): rata-rata `Close`, simpangan baku `Close`, rata-rata `Volume_USDT`
- Total 9 fitur input; target: `Close`
- Data diunduh otomatis dari Google Drive. Jika gagal, notebook memakai salinan lokal `data_raw.csv` (tidak disertakan di repo).

## Pendekatan

- **Window**: 48 jam masuk, 24 jam keluar (dipilih dari analisis ACF/PACF).
- **Split kronologis** 70% / 15% / 15% (train / validation / test), tanpa pengacakan.
- **Normalisasi Min-Max** dengan parameter dari data latih saja (tanpa data leakage).
- **Pipeline `tf.data`** untuk train, validation, dan test.
- **Model**
  - LSTM baseline (4 varian: Dense bawaan, `CustomDense`, MHA Keras, `CustomMultiHeadAttention`)
  - Seq2Seq LSTM encoder-decoder (Functional API dan Model Subclassing) dengan *teacher forcing* saat training dan inferensi *autoregressive*
- **Komponen kustom**: `CustomDense`, `CustomMultiHeadAttention`, `CustomLayerNormalization`, `CustomELU`, `CustomDropout`, loss `custom_mae` dan `horizon_weighted_mae`, `CustomEarlyStopping`, `CustomReduceLROnPlateau`.

## Hasil (data test)

| Model | MAE (scaled) | MAE (USD) |
|---|---|---|
| Naive (Close terakhir) | 0,00360 | 235,4 |
| LSTM (Dense bawaan) | 0,00396 | 259,5 |
| LSTM + CustomDense | 0,00364 | 238,6 |
| LSTM + MHA Keras | 0,00368 | 240,8 |
| **LSTM + CustomMHA (final)** | **0,00362** | 236,8 |
| Seq2Seq Functional + CustomDense | 0,00373 | 244,5 |
| Seq2Seq Subclass + MHA Keras | 0,00367 | 240,5 |
| **Seq2Seq Subclass + CustomMHA (final)** | **0,00366** | 239,3 |

Target Advanced (MAE test Seq2Seq < 0,015) tercapai.

**Catatan jujur**: pembanding naive sudah sangat kuat (0,00360). Harga Bitcoin per jam mendekati *random walk*, sehingga model belajar prediksi yang mendekati persistensi dan tidak jauh mengungguli baseline.

## Struktur Proyek

```
.
├── bitcoin_forecasting_lstm_seq2seq.ipynb  # notebook utama (EDA, training, evaluasi)
├── model_baseline_LSTM.keras           # model LSTM + CustomMHA
├── model_seq2seq_LSTM.keras            # model Seq2Seq akhir
├── best_model_seq2seq_LSTM.keras       # bobot Seq2Seq terbaik (validation)
├── requirements.txt
└── README.md
```

## Menjalankan

1. Buat environment dan pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Buka dan jalankan seluruh sel `bitcoin_forecasting_lstm_seq2seq.ipynb` secara berurutan.
3. Jumlah epoch maksimum dapat diatur lewat variabel lingkungan `DLTM_EPOCHS` (default 30).

Memuat model tersimpan: layer kustom terdaftar di notebook, jalankan sel definisi layer terlebih dahulu sebelum `keras.models.load_model(...)`.

## Dependensi

TensorFlow, Keras, NumPy, pandas, Matplotlib, seaborn, scikit-learn, statsmodels (versi pasti di `requirements.txt`).
