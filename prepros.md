### ✅ **Standardisasi Fitur RPN**

```python
scaler = StandardScaler().fit(
    pd.concat([train_df[['RPN']], val_df[['RPN']], test_df[['RPN']]])
)
```

* Anda *fit* `StandardScaler` pada gabungan seluruh data (train + val + test) untuk menghindari pergeseran skala antara set.
* Ini baik **untuk inference konsisten**, tapi **dalam praktik umum** kita hanya *fit* di `train_df` saja agar tidak terjadi *data leakage*.

  * **Best practice**:

    ```python
    scaler = StandardScaler().fit(train_df[['RPN']])
    ```

---

### ✅ **Transformasi Fitur**

```python
train_df['RPN_scaled'] = scaler.transform(train_df[['RPN']])
...
```

* `RPN_scaled` akan bernilai dengan distribusi \~𝑁(0,1). Ini cocok untuk model berbasis gradient descent seperti MLP, DNN, dsb.

---

### ✅ **Input Features**

```python
X_train = train_df[['Kategori_enc', 'RPN_scaled']].values
```

* Input terdiri dari 2 fitur numerik: encoded kategori dan scaled RPN.
* Ini akan menjadi input layer untuk model Anda (kemungkinan dense layer).

---

### ✅ **Encoding Target**

```python
le_resiko = LabelEncoder().fit(train_df['Resiko'])
y_train = to_categorical(le_resiko.transform(train_df['Resiko']))
```

* Label target `Resiko` diubah ke integer, lalu ke one-hot.
* Cocok untuk model klasifikasi multi-kelas (*softmax output*).
* Simpan `le_resiko.classes_` akan sangat membantu saat visualisasi hasil prediksi.

📌 **Tips**:

* Jika `Resiko` adalah kelas ordinal (misal: "Rendah", "Sedang", "Tinggi"), bisa pertimbangkan model klasifikasi ordinal.

---

### ✅ **Callbacks**

```python
early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
```

* Ini sangat baik, karena:

  * Early stopping mencegah overfitting.
  * Checkpoint menyimpan model terbaik berdasarkan *val\_loss*.

---
