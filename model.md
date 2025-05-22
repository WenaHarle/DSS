## 🔧 1. **Definisi Model Sequential**

```python
model = Sequential()
```

* Anda menggunakan **Keras Sequential API**, yang cocok untuk model berlapis lurus dari input ke output.

---

## 🧠 2. **Struktur Layer Jaringan Neural**

### ✨ Layer 1 (Input + Hidden pertama):

```python
model.add(Dense(128, activation='relu', input_shape=(2,)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
```

* **`Dense(128)`**: Fully connected layer dengan 128 neuron. Menerima 2 fitur (kategori dan RPN).
* **`activation='relu'`**: Fungsi aktivasi ReLU memperkenalkan non-linearitas.
* **`BatchNormalization()`**: Menstabilkan dan mempercepat training dengan menormalkan output layer sebelumnya.
* **`Dropout(0.25)`**: Mengurangi overfitting dengan menonaktifkan 25% neuron secara acak saat training.

### 📉 Hidden Layers berikutnya:

Layer-layer berikut mengulangi pola: `Dense` → `BatchNorm` → `Dropout`, dengan neuron berkurang:

```python
[96, 80, 64, 48, 32, 16, 8]
```

* Ini disebut **"tapered architecture"**, di mana neuron makin sedikit ke arah output. Biasanya digunakan untuk menyaring informasi secara bertahap dan membuat model efisien.

---

## 🎯 3. **Output Layer**

```python
model.add(Dense(y_train.shape[1], activation='softmax'))
```

* `y_train.shape[1]` adalah jumlah kelas target (jumlah kolom dari one-hot encoded `Resiko`).
* **`activation='softmax'`** membuat output menjadi distribusi probabilitas antar kelas.

---

## ⚙️ 4. **Kompilasi Model**

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

* **`optimizer='adam'`**: Optimizer adaptif berbasis kombinasi momentum dan RMSProp, cocok untuk sebagian besar kasus.
* **`loss='categorical_crossentropy'`**: Digunakan untuk klasifikasi **multi-kelas** dengan **target one-hot encoded**.
* **`metrics=['accuracy']`**: Digunakan untuk melacak akurasi selama training.

---

## 🏋️‍♂️ 5. **Training Model**

```python
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=500,
                    batch_size=32,
                    callbacks=[early_stop, checkpoint])
```

* **`X_train`, `y_train`**: Data fitur dan target pelatihan.
* **`validation_data=(X_val, y_val)`**: Digunakan untuk mengevaluasi model setiap epoch.
* **`epochs=500`**: Batas maksimum iterasi training.
* **`batch_size=32`**: Model diupdate setiap 32 sampel.
* **`callbacks=[early_stop, checkpoint]`**:

  * `EarlyStopping`: Menghentikan training lebih awal jika tidak ada peningkatan pada `val_loss` selama 100 epoch.
  * `ModelCheckpoint`: Menyimpan model terbaik berdasarkan `val_loss`.

📌 Output dari `fit()` adalah **`history`**, yang menyimpan metrik akurasi dan loss per epoch. Bisa digunakan untuk plotting.

---

## 🤖 6. **Prediksi pada Data Uji**

```python
y_pred_proba = model.predict(X_test)
```

* Menghasilkan array `y_pred_proba` berisi **probabilitas** prediksi untuk setiap kelas. Contoh:

  ```python
  [[0.1, 0.8, 0.1],
   [0.6, 0.3, 0.1], ...]
  ```

---

## 🏷️ 7. **Konversi ke Label Kelas**

```python
y_pred = np.argmax(y_pred_proba, axis=1)
```

* Mengambil **kelas dengan probabilitas tertinggi**.
* Hasilnya adalah array integer label prediksi, misalnya: `[1, 0, 2, 1, 0]`.
* Untuk interpretasi hasil:

  ```python
  label_nama = le_resiko.inverse_transform(y_pred)
  ```

---
