# Klasifikasi-Hardware
Projek klasifikasi ini merupakan Tugas Ujian Akhir Praktikum Mata Kuliah Machine Learning

## Deskripsi Proyek: Latar belakang dan tujuan pengembangan.

### Deskripsi Proyek
Proyek ini berfokus pada pengembangan sistem berbasis machine learning untuk mengklasifikasikan jenis perangkat keras. Sistem ini dirancang untuk mengenali berbagai jenis hardware dengan menganalisis karakteristik visual seperti bentuk, ukuran, dan komponen yang terlihat, sehingga dapat dikelompokan dalam kategori perangkat keras yang sesuai. 

Dataset yang digunakan pada proyek ini merupakan dataset Komponen Hardware Komputer yang diambil dari Kaggle, Dataset berjumlah 3.279 Citra. yang terdiri dari beberapa kelas yaitu
- cables 
- case               
- cpu              
- gpu                
- hdd           
- headset         
- keyboard         
- microphone            
- monitor       
- motherboard             
- mouse               
- ram          
- speakers            
- webcam 

Link Dataset : https://www.kaggle.com/datasets/gauravduttakiit/hardware-visual-dataset 

### Augmentasi 
Dataset setelah dilakukan augmentasi menjadi 10975 Citra.


### Latar Belakang 
Kemajuan teknologi telah meningkatkan kebutuhan akan sistem pintar yang mampu mendukung pengelolaan data secara efektif, termasuk dalam mengenali dan mengklasifikasikan berbagai jenis hardware komputer. Komponen seperti motherboard, proseor, RAM, hard drive, karti gratis, dan lainnya memiliki ciri visual yang khas, namun sering kali sulit dibedakan secara manual oleh pengguna yang kurang berpengalaman. 

Dengan bertambahnnya ragam perangkat hardware yang tersedia di pasaran, diperlukan solusi otomatis yang dapat mengidentifikasi dan mengelompokan hardware berdasarkan gambar visualnya. sistem semacam ini tidak hanya berguna bagi teknisi atau pengelola toko komputer, tetapi juga dimanfaatkan untuk manajemen inventaris, pelatihan teknisi pemula, hingga mendukung proses daur ulang perangkat keras yang sudah tidak terpakai.

### Tujuan Proyek
1. Mengidentifikasi berbagai jenis hardware komputer
2. Mengotomatiskan proses klasifikasi
3. Membangun dataset citra hardware yang terstruktur
4. Mempermudah manajemen inventaris hardware
5. Mendukung upaya keberlanjutan lingkungan

## Langkah Instalasi
1. Clone Repository

```bash
git clone <repository-url>
cd <repository-folder>
```
2. Buat Virtual Environment (Opsional)

```bash
python -m venv env
source env/bin/activate  # Untuk macOS/Linux
env\Scripts\activate    # Untuk Windows
```
3. Instal Dependencies

```bash
pip install -r requirements.txt
```

Jika file `requirements.txt` tidak tersedia, berikut adalah dependencies utama yang perlu diinstal:
```bash
pip install streamlit tensorflow pillow matplotlib numpy
```
4. Konfigurasi PDM (Opsional)
Jika Anda menggunakan **PDM** sebagai pengelola dependensi, inisialisasi proyek dengan:

```bash
pdm init
```
Kemudian tambahkan dependencies yang diperlukan:

```bash
pdm add streamlit tensorflow pillow matplotlib numpy
```

Untuk menjalankan aplikasi dengan PDM:

```bash
pdm run streamlit run app.py
```
5. Menempatkan Model
Pastikan Anda memiliki file model yang bernama `model.h5` dan tempatkan di folder yang sama dengan script aplikasi.

6. Jalankan Aplikasi
Jalankan aplikasi menggunakan perintah berikut:

```bash
streamlit run app.py
```

Ganti `app.py` dengan nama file Python Anda.

7. Akses Aplikasi Web
Buka browser Anda dan akses aplikasi melalui URL berikut:

```
http://localhost:8501
```

## Deskripsi Model
Pada proyek ini, model yang digunakan terdapat 2  model Deep Learning yaitu:
1. CNN (Convolutional Neural Network)
CNN adalah salah satu jenis arsitektur jaringan saraf tiruan (Neural Networks) yang dirancang khusus untuk mengolah data yang memiliki dimensi gris, seperti gambar dan video. CNN banyak digunakan dalam aplikasi computer vision, termasuk klasifikasi gambar, deteks objek, segmentasi gambar, dan lainnya. 

Contoh Arsitektur Sederhana CNN 

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Membuat model CNN
model = Sequential()

# Layer Konvolusi 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer Konvolusi 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer Fully Connected
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # Output untuk klasifikasi 10 kelas

# Menyusun model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Ringkasan model
model.summary()

```

Keunggulan CNN: 
-  Ekstraksi fitur otomatis
- Invariansi terhadap lokasi dan skala
-  Efisiensi Parameter

2. Pre-Trained MobileNet 
MobileNet adalah arsitektur deep learning yang dirancang khusus untuk perangkat dengan sumber daya terbatas seperti ponsel, tabley, dan perangkat IOT. MobileNet merupakan salah satu jenis pre-trained models dalam deep learning, yang tersedia untuk berbagai tugas di bebrerapa bidang. 

Contoh Arsitektur Sederhana MobileNet
```
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Memuat model MobileNet pre-trained pada ImageNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Menambahkan lapisan custom di atas MobileNet (untuk transfer learning)
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Mengubah output menjadi vektor satu dimensi
x = Dense(128, activation='relu')(x)  # Fully connected layer
predictions = Dense(10, activation='softmax')(x)  # Output untuk klasifikasi 10 kelas

# Membuat model akhir
model = Model(inputs=base_model.input, outputs=predictions)

# Membekukan layer pada model pre-trained agar tidak dilatih ulang
for layer in base_model.layers:
    layer.trainable = False

# Menyusun model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Ringkasan model
model.summary()

```


keunggulan MobileNet
- Ukuran model kecil
- efisiensi komputasi tinggi
dapat di transfer untuk berbagai tugas seperti klasifikasi khusus, deteksi objek, atau segmentasi.


## Hasil dan Analisis
Hasil Perbandingan Model Menggunakan CNN dan MobileNet

1. CNN 
- Accuracy pada Data Test :  0.7935
- Loss pada Data Test :  1.0231

 ### Classification Report Pada Data Test

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Cables        | 0.67      | 0.76   | 0.71     | 79      |
| Case          | 0.72      | 0.85   | 0.78     | 79      |
| CPU           | 0.90      | 0.78   | 0.84     | 79      |
| GPU           | 0.77      | 0.81   | 0.79     | 78      |
| HDD           | 0.75      | 0.75   | 0.75     | 79      |
| Headset       | 0.82      | 0.74   | 0.78     | 80      |
| Keyboard      | 0.78      | 0.81   | 0.80     | 79      |
| Microphone    | 0.79      | 0.80   | 0.79     | 79      |
| Monitor       | 0.89      | 0.78   | 0.83     | 79      |
| Motherboard   | 0.68      | 0.90   | 0.78     | 79      |
| Mouse         | 0.76      | 0.80   | 0.78     | 79      |
| RAM           | 0.90      | 0.77   | 0.83     | 79      |
| Speakers      | 0.85      | 0.76   | 0.80     | 79      |
| Webcam        | 0.97      | 0.78   | 0.87     | 79      |
| **Accuracy**  |           |        | 0.79     | 1106    |
| **Macro Avg** | 0.80      | 0.79   | 0.79     | 1106    |
| **Weighted Avg** | 0.80   | 0.79   | 0.79     | 1106    |

### Classification Report Pada Data Validasi
| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Cables        | 0.63      | 0.76   | 0.69     | 158     |
| Case          | 0.76      | 0.79   | 0.77     | 157     |
| CPU           | 0.93      | 0.84   | 0.88     | 156     |
| GPU           | 0.82      | 0.87   | 0.84     | 156     |
| HDD           | 0.76      | 0.81   | 0.78     | 157     |
| Headset       | 0.85      | 0.82   | 0.84     | 158     |
| Keyboard      | 0.72      | 0.71   | 0.72     | 157     |
| Microphone    | 0.91      | 0.79   | 0.85     | 156     |
| Monitor       | 0.88      | 0.81   | 0.84     | 156     |
| Motherboard   | 0.64      | 0.85   | 0.73     | 157     |
| Mouse         | 0.79      | 0.86   | 0.82     | 157     |
| RAM           | 0.89      | 0.74   | 0.81     | 157     |
| Speakers      | 0.89      | 0.73   | 0.81     | 158     |
| Webcam        | 0.96      | 0.87   | 0.91     | 156     |
| **Accuracy**  |           |        | 0.80     | 2196    |
| **Macro Avg** | 0.82      | 0.80   | 0.81     | 2196    |
| **Weighted Avg** | 0.82   | 0.80   | 0.81     | 2196    |

![](https://github.com/anishawlnddari/Klasifikasi-Hardware/blob/14f8f2d5ecc4e274fc817c3a331333e83f31b822/download%20(2).png)

2. MobileNet
- Accuracy pada Data Test : 0.9032
- Loss pada Data Test : 0.5952

### Classification Report Pada Data Test
| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Cables        | 0.87      | 0.82   | 0.85     | 80      |
| Case          | 0.83      | 0.90   | 0.86     | 79      |
| CPU           | 0.93      | 0.89   | 0.91     | 79      |
| GPU           | 0.93      | 0.89   | 0.91     | 79      |
| HDD           | 0.90      | 0.95   | 0.93     | 79      |
| Headset       | 0.88      | 0.89   | 0.88     | 79      |
| Keyboard      | 0.90      | 0.95   | 0.93     | 79      |
| Microphone    | 0.89      | 0.86   | 0.88     | 78      |
| Monitor       | 0.93      | 0.90   | 0.92     | 79      |
| Motherboard   | 0.90      | 0.92   | 0.91     | 79      |
| Mouse         | 0.90      | 0.92   | 0.91     | 79      |
| RAM           | 0.94      | 0.92   | 0.93     | 78      |
| Speakers      | 0.88      | 0.91   | 0.89     | 79      |
| Webcam        | 0.97      | 0.92   | 0.95     | 79      |
| **Accuracy**  |           |        | 0.90     | 1105    |
| **Macro Avg** | 0.90      | 0.90   | 0.90     | 1105    |
| **Weighted Avg** | 0.90   | 0.90   | 0.90     | 1105    |

### Classification Report Pada Data Validasi
| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Cables        | 0.80      | 0.81   | 0.81     | 158     |
| Case          | 0.82      | 0.89   | 0.86     | 156     |
| CPU           | 0.94      | 0.88   | 0.91     | 157     |
| GPU           | 0.91      | 0.94   | 0.93     | 157     |
| HDD           | 0.90      | 0.91   | 0.90     | 158     |
| Headset       | 0.92      | 0.92   | 0.92     | 157     |
| Keyboard      | 0.88      | 0.88   | 0.88     | 156     |
| Microphone    | 0.90      | 0.92   | 0.91     | 156     |
| Monitor       | 0.93      | 0.91   | 0.92     | 157     |
| Motherboard   | 0.86      | 0.87   | 0.86     | 157     |
| Mouse         | 0.93      | 0.94   | 0.93     | 157     |
| RAM           | 0.93      | 0.88   | 0.91     | 156     |
| Speakers      | 0.95      | 0.93   | 0.94     | 157     |
| Webcam        | 0.97      | 0.94   | 0.95     | 156     |
| **Accuracy**  |           |        | 0.90     | 2195    |
| **Macro Avg** | 0.90      | 0.90   | 0.90     | 2195    |
| **Weighted Avg** | 0.90   | 0.90   | 0.90     | 2195    |

![](https://github.com/anishawlnddari/Klasifikasi-Hardware/blob/51c55a26f1b40e26507e0aa82209c7ef89ed5eb6/download.png)