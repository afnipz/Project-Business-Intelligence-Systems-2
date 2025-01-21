!pip install scipy

import pandas as pd                          # Manipulasi dan analisis data berbasis DataFrame.
import numpy as np                           # Operasi numerik dan manipulasi array.

import matplotlib.pyplot as plt              # Membuat visualisasi data berupa grafik statis.
import seaborn as sns                        # Membuat visualisasi data statistik (grafik yang lebih informatif).
import plotly.express as px                  # Membuat visualisasi interaktif.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Mengubah data kategori menjadi numerik.

from sklearn.ensemble import RandomForestClassifier            # Model klasifikasi Random Forest.
from sklearn.tree import export_text, export_graphviz          # Menampilkan pohon keputusan dari model Random Forest.

from sklearn.metrics import (
    silhouette_score,                             # Evaluasi untuk clustering (silhouette score).
    accuracy_score,                               # Mengukur akurasi prediksi.
    classification_report,                        # Menampilkan metrik evaluasi lengkap (precision, recall, F1).
    confusion_matrix, ConfusionMatrixDisplay      # Membuat dan memvisualisasikan matriks kebingungan.
)

from sklearn.model_selection import (
    train_test_split,                             # Membagi data menjadi train-test.
    RandomizedSearchCV                            # Hyperparameter tuning menggunakan pencarian acak.
)

import scipy.stats as stats                       # Melakukan analisis statistik (uji distribusi, korelasi, dll).
from scipy.stats import randint                   # Membuat distribusi acak untuk hyperparameter tuning.

import graphviz                                   # Visualisasi grafis dari model pohon keputusan.

# Mendefinisikan nama file yang akan dibaca
file_path = "Churn_Modelling.csv"

# Membaca file CSV ke dalam DataFrame
df = pd.read_csv(file_path)

# Menampilkan 5 baris pertama dari DataFrame
df.head(20)

# Menampilkan dimensi DataFrame (jumlah baris dan kolom)
df.shape

# Menampilkan informasi dasar dari DataFrame
# Termasuk jumlah baris dan kolom, tipe data setiap kolom, serta jumlah nilai non-null
df.info()  # Ringkasan dasar

