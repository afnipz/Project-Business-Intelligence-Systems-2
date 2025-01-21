# Menghitung jumlah duplikasi data
num_duplicates = df.duplicated().sum()

print("Jumlah nilai duplikasi data:", num_duplicates)

# Menghapus duplikasi data dan menyimpan hasilnya
df_no_duplicates = df.drop_duplicates()

print("\nHasil setelah menghapus duplikasi data:\n")
df_no_duplicates.shape

# Memeriksa apakah kolom 'Surname' ada dalam DataFrame
if 'Surname' in df.columns: # Jika ada, hapus kolom 'Surname' menggunakan fungsi drop()
  df = df.drop('Surname', axis=1) # axis=1 menunjukkan bahwa kita menghapus kolom (bukan baris)

# Now 'Surname' column should be removed from the DataFrame
df.head()

# Memeriksa apakah kolom 'RowNumber' dan 'CustomerId' ada dalam DataFrame
if 'RowNumber' in df.columns:  # Jika 'RowNumber' ada, hapus kolom tersebut.
    df = df.drop('RowNumber', axis=1)  # axis=1 menunjukkan bahwa kita menghapus kolom (bukan baris).


if 'CustomerId' in df.columns: # Jika 'CustomerId' ada, hapus kolom tersebut.
    df = df.drop('CustomerId', axis=1) # axis=1 menunjukkan bahwa kita menghapus kolom (bukan baris).

# Sekarang kolom 'RowNumber' dan 'CustomerId' seharusnya sudah dihapus dari DataFrame.
df.head()  # Menampilkan 5 baris pertama DataFrame untuk verifikasi.

# Memeriksa nilai yang hilang di DataFrame
print("Nilai yang hilang sebelum imputasi:\n", df.isnull().sum())

# Menangani nilai yang hilang (contoh: isi dengan rata-rata untuk numerik, modus untuk kategorikal)
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

# Mengisi nilai yang hilang untuk kolom numerik dengan rata-rata
for col in numerical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# Mengisi nilai yang hilang untuk kolom kategorikal dengan modus
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# Memverifikasi apakah nilai yang hilang sudah ditangani
print("\nNilai yang hilang setelah imputasi:\n", df.isnull().sum())

# Mengidentifikasi kolom kategorikal
categorical_cols = df.select_dtypes(include=['object']).columns

# Membuat LabelEncoder
label_encoder = LabelEncoder()

# Melakukan label encoding untuk setiap kolom kategorikal
for col in categorical_cols:
    # Menyimpan mapping untuk referensi (opsional)
    mapping = dict(zip(df[col].unique(), label_encoder.fit(df[col]).transform(df[col].unique())))
    print(f"Mapping for column '{col}': {mapping}\n")

    # Melakukan encoding pada kolom yang sama
    df[col] = label_encoder.transform(df[col])

# Menampilkan DataFrame yang telah diubah
df.head()

# Loop melalui setiap kolom numerik dalam DataFrame
for column in df.select_dtypes(include=np.number):

    # Loop melalui setiap kolom numerik dalam DataFrame, kecuali kolom 'Exited'
    if column == 'Exited':
        continue

    # Buat box plot untuk kolom tersebut
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f"Box Plot untuk {column}")
    plt.show()

    # Hitung jumlah outlier menggunakan IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    num_outliers = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
    # Hapus outlier dari DataFrame
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    print(f"Jumlah outlier dalam kolom '{column}': {num_outliers}")

# Menampilkan DataFrame setelah outlier dihapus
print("\nDataFrame setelah outlier dihapus:\n")
df.head()

# Menampilkan dimensi DataFrame (jumlah baris dan kolom)
df.shape

# Simpan DataFrame yang telah di-preprocessing ke dalam file CSV
df.to_csv('update_Churn_Modelling.csv', index=False)
