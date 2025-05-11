# **Housing Price Prediction with Neural Networks**

This project builds, trains, and evaluates a neural network to predict housing prices in Polish cities based on features such as location, floor, number of rooms, and more. The model leverages regression techniques to accurately estimate property prices.

---

## **Table of Contents**
1. [Dataset Description](#dataset-description)
2. [Project Structure](#project-structure)
3. [Installation and Dependencies](#installation-and-dependencies)
4. [Running the Project](#running-the-project)
5. [Model Description](#model-description)
6. [Results and Evaluation](#results-and-evaluation)

---

## **Dataset Description**

The dataset contains housing information for properties in Polish cities with the following features:
- `latitude`: Geographic latitude of the property.
- `longitude`: Geographic longitude of the property.
- `floor`: The floor the property is located on.
- `rooms`: Number of rooms in the property.
- `sq`: Area of the property in square meters.
- `year`: Year of construction.
- `price`: Target variable representing the price of the property.

---

## **Project Structure**

```plaintext
.
├── data/                    # Processed data
│   ├── train.npy            # Training features
│   ├── test.npy             # Testing features
│   ├── train_labels.npy     # Training labels
│   ├── test_labels.npy      # Testing labels
│
├── models/                  # Saved models
│   ├── regression_model_complete.keras
│   ├── regression_model_architecture.json
│
├── outputs/                 # Evaluation results
│   ├── evaluation_summary.txt
│   ├── predictions.csv
│   ├── true_vs_predicted.png
│
├── scripts/                 # Python scripts
│   ├── data_processing.py   # Data preprocessing
│   ├── build_model.py       # Model construction
│   ├── training.py          # Model training
│   ├── evaluate_model.py    # Model evaluation
├── README.md                # Project description
├── .gitignore               # Git ignore file
└── housing_data.csv         # Git ignore file
```

---

## **Installation and Dependencies**

### **1. Clone the Repository**
```bash
git clone <repository_url>
cd <repository_directory>
```

### **2. Install Required Libraries**
Install the required Python libraries:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

---

## **Running the Project**

### **1. Data Preprocessing**
Preprocess the dataset and save the processed data:
```bash
python scripts/data_processing.py
```

### **2. Building the Model**
Construct the neural network architecture:
```bash
python scripts/build_model.py
```

### **3. Training the Model**
Train the model using the processed data:
```bash
python scripts/training.py
```

### **4. Evaluating the Model**
Evaluate the model's performance on the test set:
```bash
python scripts/evaluate_model.py
```

---

## **Model Description**

The neural network architecture is designed for regression and consists of:
- **Input Layer:** 6 input features.
- **Hidden Layers:**
  - Dense layers with 128 and 64 units, using ReLU activation.
  - Batch normalization and dropout for regularization.
- **Output Layer:** A single neuron with a linear activation function for regression output.
- **Loss Function:** Mean squared error (MSE).
- **Optimizer:** Adam optimizer.

---

## **Results and Evaluation**

### **Evaluation Metrics**
- **Mean Squared Error (MSE):** Quantifies the average squared difference between predicted and true prices.
- **Mean Absolute Error (MAE):** Measures the average magnitude of errors in predictions.

### **Visualization**
The project outputs a scatter plot comparing true prices with predicted prices, stored in the `outputs/` directory as `true_vs_predicted.png`.

### **Predictions**
The predicted and true prices are saved as a CSV file (`predictions.csv`) in the `outputs/` directory.

---

## **Notes**

- Ensure the dataset is correctly formatted and saved as `housing_data.csv` in the project directory.
- If you encounter encoding issues, use the `latin1` encoding when loading the dataset.

---

