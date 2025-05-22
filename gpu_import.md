### ✅ **GPU Setup**

```python
print("=== GPU Devices ===")
print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

* Menampilkan GPU yang terdeteksi oleh TensorFlow.
* Menghindari alokasi memori GPU penuh dengan `set_memory_growth(gpu, True)`.

📌 **Catatan**:

* `tf.config.experimental.set_memory_growth()` adalah **experimental API**; jika Anda menggunakan TensorFlow ≥ 2.1, disarankan untuk menggunakan:

  ```python
  tf.config.set_logical_device_configuration(
      gpu,
      [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
  )
  ```

  jika ingin membatasi memori tertentu.

---

### ✅ **Load CSV Dataset**

```python
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')
```

* Memuat tiga dataset: pelatihan, validasi, dan pengujian.

---

### ✅ **Label Encoding**

```python
all_kategori = pd.concat([train_df['Kategori'], val_df['Kategori'], test_df['Kategori']])
le_kat = LabelEncoder().fit(all_kategori)

train_df['Kategori_enc'] = le_kat.transform(train_df['Kategori'])
val_df['Kategori_enc'] = le_kat.transform(val_df['Kategori'])
test_df['Kategori_enc'] = le_kat.transform(test_df['Kategori'])
```

* Ini penting untuk memastikan label encoding konsisten di seluruh dataset.
* `LabelEncoder` dilatih pada gabungan seluruh label kategori agar **semua kelas dikenali**, menghindari `ValueError` saat transformasi.

---
