## 📦 **1. Import Library untuk Manipulasi Data**

```python
import pandas as pd
import numpy as np
```

* `pandas (pd)`:

  * Digunakan untuk **manipulasi dan analisis data berbasis tabel** (DataFrame).
  * Contoh penggunaan: membaca file `.csv`, menampilkan data, filter baris/kolom, groupby, dll.
  * Contoh: `df = pd.read_csv("data.csv")`

* `numpy (np)`:

  * Library untuk **komputasi numerik dan array**.
  * Sangat efisien untuk operasi matematis pada array besar.
  * Contoh: `np.array`, `np.mean`, `np.std`, operasi matriks.

---

## 📊 **2. Import Library untuk Visualisasi Data**

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

* `matplotlib.pyplot (plt)`:

  * Library dasar untuk membuat **grafik dan plot**.
  * Contoh plot: garis, histogram, scatter plot.
  * Contoh: `plt.plot(x, y)`, `plt.show()`

* `seaborn (sns)`:

  * Library visualisasi berbasis matplotlib.
  * Fokus pada visualisasi statistik seperti heatmap, boxplot, pairplot, dll.
  * Lebih estetik dan cocok untuk eksplorasi data.
  * Contoh: `sns.heatmap(confusion_matrix)`

---

## 🛠️ **3. Import Library untuk Preprocessing (Persiapan Data)**

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

* `LabelEncoder`:

  * Mengubah **label kategori menjadi angka**.
  * Contoh: `"cat", "dog", "mouse"` → `0, 1, 2`.

* `StandardScaler`:

  * Melakukan **standarisasi fitur**:

    * Mean = 0, Std = 1.
  * Membantu model konvergen lebih cepat dan stabil.

---

## 📏 **4. Import Metrik Evaluasi Model**

```python
from sklearn.metrics import classification_report, confusion_matrix
```

* `classification_report`:

  * Menampilkan metrik evaluasi klasifikasi seperti **precision, recall, f1-score, dan support** untuk tiap kelas.

* `confusion_matrix`:

  * Menampilkan matriks **benar/salah prediksi** model:

    * Baris = kelas aktual.
    * Kolom = kelas prediksi.

---

## 🤖 **5. Import Library Deep Learning (TensorFlow dan Keras)**

```python
import tensorflow as tf
```

* TensorFlow adalah framework deep learning dari Google.
* `tf.keras` adalah antarmuka Keras dalam TensorFlow untuk membangun dan melatih model neural network.

---

## 🧱 **6. Import Model dan Layer Keras**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
```

* `Sequential`:

  * Model neural network **linear (berlapis bertumpuk)**.
  * Setiap layer disusun secara berurutan.

* `Dense`:

  * Fully Connected Layer.
  * Setiap neuron terhubung ke semua neuron di layer sebelumnya.

* `Dropout`:

  * Teknik regularisasi untuk **mengurangi overfitting**.
  * Randomly mematikan sejumlah neuron saat training.

---

## 🛑 **7. Import Callback untuk Mengatur Proses Pelatihan**

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```

* `EarlyStopping`:

  * Menghentikan training **jika tidak ada peningkatan** pada `val_loss` atau `val_accuracy` setelah beberapa epoch.

* `ModelCheckpoint`:

  * Menyimpan **model terbaik secara otomatis** selama training.

---

## 🔢 **8. Import Utility Keras untuk One-Hot Encoding**

```python
from tensorflow.keras.utils import to_categorical
```

* `to_categorical`:

  * Mengubah label numerik menjadi bentuk **one-hot encoding**.
  * Contoh: `2` → `[0, 0, 1]`.

---

## ⚖️ **9. Import Layer Tambahan untuk Stabilisasi Training**

```python
from tensorflow.keras.layers import BatchNormalization
```

* `BatchNormalization`:

  * Menormalkan aktivasi layer selama training.
  * Membantu mengatasi **vanishing gradient** dan mempercepat training.

---

## 🛡️ **10. Import Regularizer**

```python
from tensorflow.keras import regularizers
```

* `regularizers`:

  * Untuk menambahkan **L1 / L2 regularisasi** pada bobot layer.
  * Membantu **menghindari overfitting** dengan menghukum bobot besar.
  * Contoh: `kernel_regularizer=regularizers.l2(0.01)`

---
