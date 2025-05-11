import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import itertools
import matplotlib.pyplot as plt

# Sciezki
model_dir = "./models"
data_dir = "./data"
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)  # Upewnij sie, ze katalog wyjsciowy istnieje

# Sciezki do wynikow (dolaczyc do sprawozdania pliki graf)
tuning_results_path = os.path.join(output_dir, "hyperparameter_tuning_results.csv")
tuning_plot_path = os.path.join(output_dir, "hyperparameter_tuning_plot.png")
epoch_analysis_path = os.path.join(output_dir, "epoch_analysis.csv")
epoch_plot_path = os.path.join(output_dir, "epoch_analysis_plot.png")
training_history_path = os.path.join(output_dir, "training_history.csv")
learning_curve_path = os.path.join(output_dir, "learning_curve.png")
model_architecture_path = os.path.join(output_dir, "model_architecture.png")

# Wczytaj dane treningowe
X_train = np.load(os.path.join(data_dir, "train.npy"))
y_train = np.load(os.path.join(data_dir, "train_labels.npy"))

# Wczytaj model
model = load_model(os.path.join(model_dir, "dense_model_complete.keras"))

# Generuj i zapisz wizualizacje architektury modelu
plot_model(model, to_file=model_architecture_path, show_shapes=True)
print(f"Wizualizacja architektury modelu zapisana do {model_architecture_path}.")

# Callbacks do lepszego treningu
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

# Strojenie hiperparametrow
batch_sizes = [16, 32, 64]
learning_rates = [0.001, 0.0005, 0.0001]
optimizers = {
    'adam': Adam,
    'sgd': SGD,
    'rmsprop': RMSprop
}
results = []

for batch_size, lr, opt_name in itertools.product(batch_sizes, learning_rates, optimizers.keys()):
    print(f"Trening z batch_size={batch_size}, learning_rate={lr}, optimizer={opt_name}")
    optimizer = optimizers[opt_name](learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    final_val_loss = history.history['val_loss'][-1]
    results.append({
        'batch_size': batch_size,
        'learning_rate': lr,
        'optimizer': opt_name,
        'val_loss': final_val_loss
    })

results_df = pd.DataFrame(results)
results_df.to_csv(tuning_results_path, index=False)
print(f"Wyniki strojenia hiperparametrow zapisane do {tuning_results_path}.")

# Wykres wynikow strojenia hiperparametrow
plt.figure(figsize=(10, 5))
for opt_name in results_df['optimizer'].unique():
    subset = results_df[results_df['optimizer'] == opt_name]
    plt.plot(subset['learning_rate'], subset['val_loss'], marker='o', label=f"Optimizer: {opt_name}")
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.title('Strojenie Hiperparametrow: Validation Loss vs Learning Rate')
plt.legend()
plt.savefig(tuning_plot_path)
plt.show()

# Analiza epok
epochs_to_test = [50, 100, 200]
epoch_results = []

for num_epochs in epochs_to_test:
    print(f"Trening z {num_epochs} epokami.")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=num_epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    best_val_loss = min(history.history['val_loss'])
    epoch_results.append({'epochs': num_epochs, 'best_val_loss': best_val_loss})

epoch_results_df = pd.DataFrame(epoch_results)
epoch_results_df.to_csv(epoch_analysis_path, index=False)
print(f"Wyniki analizy epok zapisane do {epoch_analysis_path}.")

# Wykres analizy epok
plt.figure(figsize=(10, 5))
plt.plot(epoch_results_df['epochs'], epoch_results_df['best_val_loss'], marker='o')
plt.xlabel('Liczba Epok')
plt.ylabel('Najlepszy Validation Loss')
plt.title('Analiza Epok: Validation Loss vs Liczba Epok')
plt.savefig(epoch_plot_path)
plt.show()

# Finalny trening i zapis modelu
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

model.save(os.path.join(model_dir, "dense_model_complete.keras"))
print(f"Przeszkolony model zapisany do {os.path.join(model_dir, 'dense_model_complete.keras')}.")

# Eksport finalnej historii treningu
history_df = pd.DataFrame(history.history)
history_df.to_csv(training_history_path, index=False)
print(f"Historia treningu zapisana do {training_history_path}.")

# Wykres finalnej krzywej uczenia
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoki')
plt.ylabel('Loss')
plt.title('Krzywa Uczenia')
plt.legend()
plt.savefig(learning_curve_path)
plt.show()
print(f"Krzywa uczenia zapisana do {learning_curve_path}.")
