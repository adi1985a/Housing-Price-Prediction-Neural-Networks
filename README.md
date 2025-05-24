# 🏠 Housing Price Prediction with Neural Networks

![Housing Price Prediction](https://via.placeholder.com/800x200.png?text=Housing+Price+Prediction)  
*Predict property prices in Polish cities using neural networks*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Platform: Any](https://img.shields.io/badge/Platform-Any-lightgrey.svg)](https://www.python.org/downloads/)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset Description](#dataset-description)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Description](#model-description)
- [Results and Evaluation](#results-and-evaluation)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Overview
This project develops a neural network to predict housing prices in Polish cities, leveraging features such as location, floor, number of rooms, and property size. Built with TensorFlow, the model uses regression techniques to deliver accurate price estimates, making it a valuable tool for real estate analysis. The project includes data preprocessing, model training, evaluation, and visualization of results, all implemented in Python.

## Features
- **🔍 Accurate Predictions**: Uses a neural network for precise housing price regression.
- **📊 Data Processing**: Handles geographic, structural, and temporal property features.
- **🛠️ Modular Scripts**: Separate scripts for preprocessing, model building, training, and evaluation.
- **📈 Visualization**: Generates scatter plots comparing true vs. predicted prices.
- **💾 Output Storage**: Saves predictions and evaluation metrics for further analysis.

## Dataset Description
The dataset includes housing data from Polish cities with the following features:
- **latitude**: Geographic latitude of the property.
- **longitude**: Geographic longitude of the property.
- **floor**: The floor the property is located on.
- **rooms**: Number of rooms in the property.
- **sq**: Area of the property in square meters.
- **year**: Year of construction.
- **price**: Target variable representing the property price.

The dataset is provided as `housing_data.csv` and is processed into NumPy arrays for training and testing.

## Screenshots
*Coming soon!*

## Project Structure

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

