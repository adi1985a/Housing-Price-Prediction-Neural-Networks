import os

# Przetwarzanie danych
os.system("python scripts/data_processing.py")

# Budowa modelu
os.system("python scripts/build_model.py")

# Trening modelu
os.system("python scripts/training.py")

# Ewaluacja modelu
os.system("python scripts/evaluate_model.py")
