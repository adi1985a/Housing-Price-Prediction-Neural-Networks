import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

# Sciezki folderow
data_dir = "./data"
outputs_dir = "./outputs"
models_dir = "./models"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

train_data_path = os.path.join(data_dir, "train.npy")
test_data_path = os.path.join(data_dir, "test.npy")
train_labels_path = os.path.join(data_dir, "train_labels.npy")
test_labels_path = os.path.join(data_dir, "test_labels.npy")

# Wczytaj dane
data = pd.read_csv("housing_data.csv", encoding="latin1")

# Wybierz cechy i cel
features = ['latitude', 'longitude', 'floor', 'rooms', 'sq', 'year']
target = 'price'

# Walidacja
assert data[features].isnull().sum().sum() == 0, "Cechy maja brakujace wartosci!"
assert data[target].isnull().sum() == 0, "Cel ma brakujace wartosci!"
assert (data[features] >= 0).all().all(), "Niektore cechy maja ujemne wartosci!"

# Usun brakujace wartosci
data = data.dropna()

# Wykres rozkladu celu przed usunieciem wartosci odstajacych
plt.figure(figsize=(10, 6))
sns.histplot(data[target], bins=50, kde=True, color='skyblue')
plt.title("Rozklad celu (price) przed usunieciem wartosci odstajacych")
plt.xlabel("Cena")
plt.ylabel("Ilosc")
plt.savefig(os.path.join(outputs_dir, "price_distribution_before_outliers.png"))
plt.show()

# Usun wartosci odstajace
data = data[data[target] < data[target].quantile(0.99)]

# Wykres rozkladu celu po usunieciu wartosci odstajacych
plt.figure(figsize=(10, 6))
sns.histplot(data[target], bins=50, kde=True, color='orange')
plt.title("Rozklad celu (price) po usunieciu wartosci odstajacych")
plt.xlabel("Cena")
plt.ylabel("Ilosc")
plt.savefig(os.path.join(outputs_dir, "price_distribution_after_outliers.png"))
plt.show()

# Cechy i cel
X = data[features].values
y = data[target].values

# Skalowanie
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Zapisz skalery
dump(scaler_X, os.path.join(models_dir, "scaler_X.joblib"))
dump(scaler_y, os.path.join(models_dir, "scaler_y.joblib"))

# Podziel dane
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Zapisz dane
np.save(train_data_path, X_train)
np.save(test_data_path, X_test)
np.save(train_labels_path, y_train)
np.save(test_labels_path, y_test)

# Zapisz statystyki opisowe
descriptive_stats = data[features + [target]].describe()
descriptive_stats.to_csv(os.path.join(data_dir, "descriptive_stats.csv"))

# Heatmapa korelacji
plt.figure(figsize=(10, 8))
correlation_matrix = data[features + [target]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Mapa korelacji")
plt.savefig(os.path.join(outputs_dir, "correlation_heatmap.png"))
plt.show()

# Augmentacja danych
augmented_data = pd.DataFrame(X_train, columns=features)
augmented_data['latitude'] += np.random.normal(0, 0.01, size=augmented_data.shape[0])
augmented_data['longitude'] += np.random.normal(0, 0.01, size=augmented_data.shape[0])
augmented_data['floor'] = augmented_data['floor'] * np.random.uniform(0.9, 1.1, size=augmented_data.shape[0])

X_train_augmented = np.vstack((X_train, augmented_data.values))
y_train_augmented = np.hstack((y_train, y_train))

# Zapisz dane po augmentacji
np.save(os.path.join(data_dir, "train_augmented.npy"), X_train_augmented)
np.save(os.path.join(data_dir, "train_labels_augmented.npy"), y_train_augmented)

# Zapisz losowe probki
random_samples = X_train[np.random.choice(X_train.shape[0], 3, replace=False)]
random_samples_df = pd.DataFrame(random_samples, columns=features)
random_samples_df['price'] = y_train[np.random.choice(y_train.shape[0], 3, replace=False)]
random_samples_df.to_csv(os.path.join(data_dir, "random_samples.csv"), index=False)

# Histogramy dla cech
for feature in features:
    plt.figure(figsize=(8, 6))
    sampled_data = data[feature].sample(n=1000, random_state=42) if feature in ['sq', 'year'] else data[feature]
    sns.histplot(sampled_data, kde=True, color='blue')
    plt.title(f'Histogram i wykres gestosci dla {feature}')
    plt.savefig(os.path.join(outputs_dir, f"{feature}_histogram.png"))
    plt.show()

# Raport podsumowujacy
report_path = os.path.join(data_dir, "data_report.txt")
with open(report_path, "w") as report:
    report.write("Podsumowanie danych:\n")
    report.write(str(descriptive_stats))
    report.write("\n\nMapa korelacji zapisana jako 'correlation_heatmap.png'.\n")
    report.write("Dane po augmentacji zapisane jako 'train_augmented.npy' i 'train_labels_augmented.npy'.\n")
    report.write("Losowe probki zapisane w 'random_samples.csv'.")
