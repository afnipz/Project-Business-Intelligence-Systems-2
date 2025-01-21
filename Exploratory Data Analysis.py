df_updated = pd.read_csv('update_Churn_Modelling.csv')
df_updated.head()  # Menampilkan beberapa baris pertama dari data yang telah diupdate

df_updated.columns

# Korelasi antar fitur dengan target (Exited)
correlation_matrix = df_updated.corr()

# Heatmap untuk melihat korelasi
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasi antar Fitur')
plt.show()

# Fokus pada korelasi dengan kolom target 'Exited'
correlation_with_target = correlation_matrix['Exited'].sort_values(ascending=False)

print("Korelasi dengan target (Exited):\n", correlation_with_target)

# Mengatur palet warna
sns.set_palette("pastel")

# Pisahkan fitur numerik
numerical_features = df_updated.select_dtypes(include=np.number).columns.tolist()

# Memplot histogram
fig, axes = plt.subplots(11, 1, figsize=(10, 48))  # Sesuaikan jumlah subplot dengan jumlah fitur numerik
axes = axes.flatten()

# Iterasi dan Plot Histogram
for i, feature in enumerate(numerical_features):
    if i < len(axes):  # Pastikan index tidak melebihi batas axes
        hist_plot = sns.histplot(df_updated[feature], bins=15, kde=False, ax=axes[i], color=sns.color_palette("pastel")[i % len(sns.color_palette("pastel"))])
        axes[i].set_title(feature, fontsize=12)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

        # Menambahkan keterangan angka di atas setiap bar
        for patch in hist_plot.patches:
            height = patch.get_height()
            if height > 0:  # Jika ada nilai
                axes[i].text(
                    patch.get_x() + patch.get_width() / 2,
                    height + 0.5,  # Posisi sedikit di atas bar
                    int(height),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black"
                )
    else:
        break

# Menyusun tata letak dan menampilkan judul utama
plt.tight_layout()
plt.suptitle('Histogram', fontsize=16, y=1.02)
plt.show()

# Scatter Plot untuk melihat hubungan antara CreditScore dan Balance
plt.figure(figsize=(8, 6))
sns.scatterplot(x='CreditScore', y='Balance', data=df_updated, hue='Exited')
plt.title('Hubungan antara CreditScore dan Balance')
plt.xlabel('CreditScore')
plt.ylabel('Balance')
plt.show()

# Scatter Plot untuk melihat hubungan antara Age dan EstimatedSalary
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='EstimatedSalary', data=df_updated, hue='Exited')
plt.title('Hubungan antara Age dan EstimatedSalary')
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.show()

# Scatter Plot untuk melihat hubungan antara Tenure dan Balance
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Tenure', y='Balance', data=df_updated, hue='Exited')
plt.title('Hubungan antara Tenure dan Balance')
plt.xlabel('Tenure')
plt.ylabel('Balance')
plt.show()
