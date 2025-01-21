# Pisahkan fitur dan target dari dataset
# df_updated adalah dataset yang telah diproses sebelumnya
X = df_updated.drop('Exited', axis=1)  # Fitur
y = df_updated['Exited']  # Target

# List untuk menyimpan hasil pengujian
results = []

# Daftar ukuran test set dan jumlah fitur untuk diuji
test_sizes = [0.1, 0.2, 0.3, 0.4]
feature_counts = [3, 4, 5]

# Pengujian dengan berbagai kombinasi
for test_size in test_sizes:
    for feature_count in feature_counts:
        # Latih model Random Forest awal untuk menentukan fitur penting
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)

        # Ambil daftar fitur penting berdasarkan importance
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Pilih fitur dengan importance tertinggi
        selected_features = feature_importances['Feature'][:feature_count]
        X_selected = X[selected_features]

        # Split data dengan ukuran test set yang ditentukan
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42
        )

        # Latih model Random Forest
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Hitung akurasi
        accuracy = accuracy_score(y_test, y_pred)

        # Simpan hasil pengujian
        results.append({
            'Test Size': test_size,
            'Feature Count': feature_count,
            'Accuracy': accuracy
        })

# Tampilkan hasil pengujian
results_df = pd.DataFrame(results)
print(results_df)

# Cari hasil terbaik
best_result = results_df.loc[results_df['Accuracy'].idxmax()]
print("\nBest Result:")
print(best_result)

# Split data dengan ukuran test set tetap (10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Latih Random Forest untuk menentukan fitur penting
rf_temp = RandomForestClassifier(random_state=42)
rf_temp.fit(X_train, y_train)

# Pilih 5 fitur terpenting
feature_importances = pd.Series(rf_temp.feature_importances_, index=X_train.columns).sort_values(ascending=False)
important_features = list(feature_importances.nlargest(3).index)
print("Top 5 features selected:", important_features)

# Data hanya menggunakan fitur yang dipilih
X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

# Definisi distribusi parameter
param_dist = {'n_estimators': randint(100, 500),
              'max_depth': randint(5, 20)}

# Randomized search
rand_search = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                 param_distributions=param_dist,
                                 n_iter=5,
                                 cv=5,
                                 random_state=42)
rand_search.fit(X_train_selected, y_train)

# Hasil hyperparameter terbaik
best_params = rand_search.best_params_
print("Best hyperparameters:", best_params)

# Model terbaik
best_rf_model = rand_search.best_estimator_
y_pred = best_rf_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the best model is {accuracy:.5f}")

# Simpan skor dari setiap pohon dalam Random Forest
scores = [tree.score(X_train_selected, y_train) for tree in best_rf_model.estimators_]
best_tree_index = np.argmax(scores)
best_tree = best_rf_model.estimators_[best_tree_index]
print(f"Best tree index: {best_tree_index}")

# Visualisasi aturan pohon terbaik
dot_data = export_graphviz(best_tree,
                           feature_names=X_train_selected.columns,
                           filled=True,
                           class_names=['Not Exited', 'Exited'],
                           max_depth=3,
                           impurity=False,
                           proportion=True)
graph = graphviz.Source(dot_data)
display(graph)

# Aturan dalam bentuk teks
tree_rules = export_text(best_tree, feature_names=list(X_train_selected.columns), max_depth=2)
print("Aturan dari pohon terbaik hingga kedalaman tertentu:")
print(tree_rules)

