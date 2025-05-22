## 🔹 `Dense(128)`

### ➤ Apa itu?

Layer `Dense` (fully connected) adalah lapisan di mana **setiap neuron** terhubung ke **seluruh neuron** di layer sebelumnya.

### ➤ Parameter `128`

Artinya layer ini memiliki **128 neuron**. Masing-masing neuron akan:

* Menerima **dua input** (karena `input_shape=(2,)`)
* Melakukan transformasi linier:

  $$
  z = w_1 \cdot x_1 + w_2 \cdot x_2 + b
  $$

  untuk setiap neuron.

### ➤ Fungsi:

* Mempelajari **representasi fitur yang lebih kompleks** dari input.
* Memberi fleksibilitas tinggi pada model dalam melakukan pemetaan input ke output.

---

## 🔹 `activation='relu'`

### ➤ Apa itu ReLU?

**ReLU (Rectified Linear Unit)** adalah fungsi aktivasi yang didefinisikan sebagai:

$$
f(x) = \max(0, x)
$$

### ➤ Fungsi Aktivasi?

* Memperkenalkan **non-linearitas** dalam jaringan saraf.
* Tanpa aktivasi non-linear, jaringan hanya dapat memodelkan hubungan linier.

### ➤ Kenapa ReLU?

* **Sederhana dan cepat dihitung**.
* Mengatasi masalah **vanishing gradient** (dibanding sigmoid/tanh).
* Cocok untuk deep neural networks.

---

## 🔹 `BatchNormalization()`

### ➤ Apa itu?

Batch Normalization adalah teknik untuk **menormalkan output layer sebelumnya**, dengan cara:

1. Mengurangi **mean** dan membagi dengan **standar deviasi** per batch:

   $$
   \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
   $$
2. Kemudian dilakukan transformasi skala dan pergeseran:

   $$
   y = \gamma \hat{x} + \beta
   $$

### ➤ Tujuan:

* **Menstabilkan dan mempercepat pelatihan**.
* Mengurangi sensitivitas terhadap inisialisasi bobot.
* Bertindak seperti regularisasi ringan (terutama jika tanpa dropout).

### ➤ Kapan dipakai?

* Biasanya digunakan **setelah layer Dense dan sebelum aktivasi**, tapi pada praktik umum bisa juga setelah aktivasi — tergantung eksperimen.

---

## 🔹 `Dropout(0.25)`

### ➤ Apa itu Dropout?

Dropout adalah teknik regularisasi yang **menonaktifkan sejumlah neuron secara acak** selama training.

### ➤ Angka 0.25:

* Menonaktifkan **25% neuron secara acak** dalam batch selama proses training.
* Pada inference/prediksi, dropout **tidak diaktifkan**.

### ➤ Tujuan:

* **Mengurangi overfitting**: mencegah neuron terlalu bergantung satu sama lain (co-adaptation).
* Mendorong jaringan untuk belajar **representasi yang lebih robust dan umum**.

### ➤ Analogi:

* Seperti "membuang" sebagian neuron sementara agar jaringan belajar dengan lebih tahan terhadap kehilangan informasi.

---

## 🔁 Kombinasi Ketiganya:

### Secara urut:

```python
Dense(128, activation='relu') 
→ BatchNormalization() 
→ Dropout(0.25)
```

* **Dense + ReLU**: membentuk representasi baru dari fitur.
* **BatchNormalization**: menstabilkan dan mempercepat training.
* **Dropout**: mencegah model terlalu menghafal data (overfitting).

---

## 📊 Visualisasi Alur

Misalnya input: `[Kategori_enc, RPN_scaled]` → masuk ke layer pertama:

1. **Input (2 nilai)** →
2. **Dense(128)** → hasil transformasi linier dari 2 input jadi 128 output neuron
3. **ReLU** → semua nilai negatif jadi 0
4. **BatchNormalization** → output dinormalisasi (mean \~0, std \~1)
5. **Dropout(0.25)** → 25% dari neuron dimatikan sementara

---
