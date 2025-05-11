import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance

# Klasa wrappera dla kompatybilności ze scikit-learn
class KerasRegressorWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        pass

    def score(self, X, y):
        predictions = self.model.predict(X).flatten()
        return -mean_squared_error(y, predictions)

# Konfiguracja ścieżek
data_dir = "./data"
outputs_dir = "./outputs"
models_dir = "./models"
os.makedirs(outputs_dir, exist_ok=True)

# Funkcja do zapisywania plików tekstowych
def save_text(filepath, content):
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)

# Funkcja do tworzenia i zapisywania wykresów
def save_plot(fig, filepath):
    fig.savefig(filepath)
    plt.close(fig)

# Wczytaj dane i model
def load_data_and_model():
    model_path = os.path.join(models_dir, "dense_model_complete.keras")
    X_test = np.load(os.path.join(data_dir, "test.npy"))
    y_test = np.load(os.path.join(data_dir, "test_labels.npy"))
    model = load_model(model_path)
    return model, X_test, y_test

# Ewaluacja modelu
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, mae, r2

# Generowanie raportu z ewaluacji
def generate_evaluation_report(mse, mae, r2):
    content = f"Mean Squared Error (MSE): {mse:.4f}\n"
    content += f"Mean Absolute Error (MAE): {mae:.4f}\n"
    content += f"R^2 Score: {r2:.4f}\n"
    save_text(os.path.join(outputs_dir, "evaluation_summary.txt"), content)

# Zapisz przewidywania
def save_predictions(y_test, y_pred):
    predictions_df = pd.DataFrame({"True Price": y_test, "Predicted Price": y_pred})
    predictions_df.to_csv(os.path.join(outputs_dir, "predictions.csv"), index=False)

# Wykres prawdziwe vs przewidywane wartości
def plot_true_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, label="Predykcja")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Idealne Dopasowanie")
    ax.set_xlabel('Prawdziwe Ceny')
    ax.set_ylabel('Przewidywane Ceny')
    ax.set_title('Prawdziwe vs Przewidywane Ceny')
    ax.legend()
    plt.show()
    save_plot(fig, os.path.join(outputs_dir, "true_vs_predicted.png"))

# Wykres rozkładu błędów
def plot_error_distribution(errors):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(errors, kde=True, color="blue", ax=ax)
    ax.set_title("Rozkład Błędów")
    ax.set_xlabel("Błąd")
    ax.set_ylabel("Ilość")
    plt.show()
    save_plot(fig, os.path.join(outputs_dir, "error_distribution.png"))

# Zapisz statystyki błędów
def save_error_statistics(errors):
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    content = f"Średni Błąd: {mean_error:.4f}\nOdchylenie Standardowe Błędu: {std_error:.4f}\n"
    save_text(os.path.join(outputs_dir, "error_statistics.txt"), content)

# Zapisz największe błędy
def save_largest_errors(y_test, y_pred, errors):
    errors_df = pd.DataFrame({"True Price": y_test, "Predicted Price": y_pred, "Error": np.abs(errors)})
    largest_errors = errors_df.sort_values(by="Error", ascending=False).head(10)
    largest_errors.to_csv(os.path.join(outputs_dir, "largest_errors.csv"), index=False)

# Generowanie analizy SHAP
def generate_shap_analysis(model, X_test):
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.show()
    save_plot(plt.gcf(), os.path.join(outputs_dir, "feature_importance.png"))

# Ważność cech za pomocą permutacji
def generate_permutation_importance(model, X_test, y_test):
    wrapped_model = KerasRegressorWrapper(model)
    perm_importance = permutation_importance(wrapped_model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importances = pd.DataFrame({
        "Feature": [f"Feature {i}" for i in range(X_test.shape[1])],
        "Importance": perm_importance.importances_mean
    }).sort_values(by="Importance", ascending=False)
    feature_importances.to_csv(os.path.join(outputs_dir, "feature_importances.csv"), index=False)

# Główna funkcja pipeline
def main():
    model, X_test, y_test = load_data_and_model()
    y_pred, mse, mae, r2 = evaluate_model(model, X_test, y_test)

    # Zapisz wyniki ewaluacji
    generate_evaluation_report(mse, mae, r2)
    save_predictions(y_test, y_pred)

    # Wykresy wyników
    plot_true_vs_predicted(y_test, y_pred)
    errors = y_test - y_pred
    plot_error_distribution(errors)

    # Zapisz analizę błędów
    save_error_statistics(errors)
    save_largest_errors(y_test, y_pred, errors)

    # Generowanie ważności cech
    generate_shap_analysis(model, X_test)
    generate_permutation_importance(model, X_test, y_test)

if __name__ == "__main__":
    main()
