import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Sciezki do folderow
models_dir = "./models"
logs_dir = "./logs"
visualizations_dir = "./outputs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

# Sciezki do plikow modelu
dense_architecture_path = os.path.join(models_dir, "dense_model_architecture.json")
dense_model_path = os.path.join(models_dir, "dense_model_complete.keras")
conv1d_architecture_path = os.path.join(models_dir, "conv1d_model_architecture.json")
conv1d_model_path = os.path.join(models_dir, "conv1d_model_complete.keras")

# Funkcja do budowy modeli regresji
def build_model(architecture="dense"):
    if architecture == "dense":
        model = Sequential([
            Dense(256, activation='relu', input_dim=6),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='linear')  # Dla wynikow regresji
        ])
    elif architecture == "conv1d":
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(6, 1)),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='linear')  # Dla wynikow regresji
        ])
    else:
        raise ValueError("Nieobsugiwany typ architektury")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',  # Alternatywy: 'mean_absolute_error', 'huber' / porownac wyniki, ktory model daje lepsze wyniki
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
    return model

# Inicjalizacja modelu Dense
model_dense = build_model(architecture="dense")

# Zapisz architekture Dense do JSON
with open(dense_architecture_path, "w") as file:
    file.write(model_dense.to_json())

# Zapisz nieprzeszkolony model Dense
model_dense.save(dense_model_path)

# Wyswietl podsumowanie modelu Dense
print("Architektura Dense:")
model_dense.summary()

# Inicjalizacja modelu Conv1D
model_conv1d = build_model(architecture="conv1d")

# Zapisz architekture Conv1D do JSON
with open(conv1d_architecture_path, "w") as file:
    file.write(model_conv1d.to_json())

# Zapisz nieprzeszkolony model Conv1D
model_conv1d.save(conv1d_model_path)

# Wyswietl podsumowanie modelu Conv1D
print("\nArchitektura Conv1D:")
model_conv1d.summary()
