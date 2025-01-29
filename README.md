# Forecasting Oil, Water, and Gas Production Using Machine Learning

## 📌 Overview
This repository contains Jupyter Notebooks demonstrating the application of machine learning models for predicting oil, water, and gas production based on wellhead parameters. The models implemented include:

- **Recurrent Neural Networks (RNN) & Long Short-Term Memory (LSTM)**
- **Decision Tree Regressor (DTR) & Random Forest Regressor (RFR)**

Each notebook provides a step-by-step guide on preprocessing data, training models, and evaluating their performance.

---

## 📝 Dataset Description
The dataset used in this study consists of production-related parameters:

- **Inputs:**
  - Wellhead Pressure
  - Annular Pressure
  - Downhole Temperature
  - Downhole Pressure
  - Choke Size
  - Production Date

- **Outputs:**
  - Oil Production
  - Water Production
  - Gas Production

---

## 🔍 Machine Learning Models

### 1️⃣ RNN & LSTM Model (`RNN_LSTM_BY_SUNUSI.ipynb`)
This notebook explores **sequence modeling** using RNN and LSTM for predicting production rates.

#### **Key Steps**
✅ Importing necessary libraries  
✅ Loading and preprocessing the dataset  
✅ Constructing **RNN & LSTM** models using TensorFlow/Keras  
✅ Training and evaluating the models  

#### **Code Snippet**
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sample model structure
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
```

---

### 2️⃣ Decision Tree & Random Forest Regressor (`DTR_RFR_BY_SUNUSI.ipynb`)
This notebook applies **Decision Tree Regressor (DTR) and Random Forest Regressor (RFR)** to forecast oil, water, and gas production.

#### **Key Steps**
✅ Importing necessary libraries  
✅ Feature engineering and dataset preparation  
✅ Implementing **DTR and RFR models** using Scikit-learn  
✅ Model evaluation and comparison  

#### **Code Snippet**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Initializing models
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor(n_estimators=100)

# Training
dtr.fit(X_train, y_train)
rfr.fit(X_train, y_train)

# Predictions
y_pred_dtr = dtr.predict(X_test)
y_pred_rfr = rfr.predict(X_test)

# Evaluation
print("DTR MSE:", mean_squared_error(y_test, y_pred_dtr))
print("RFR MSE:", mean_squared_error(y_test, y_pred_rfr))
```

---

## 📊 Results & Model Comparison

| Model  | Target | R² Score | Mean Squared Error (MSE) | Mean Absolute Error (MAE) |
|--------|--------|----------|--------------------------|--------------------------|
| **LSTM** | **Oil**   | *(0.950 )* | *( 0.004)* | *(0.030 )* |
|          | **Gas**   | *( 0.950)* | *(0.004 )* | *(0.030 )* |
|          | **Water** | *( 0.950)* | *( 0.002)* | *( 0.026)* |
| **RNN**  | **Oil**   | *(0.930 )* | *( 0.004)* | *(0.032 )* |
|          | **Gas**   | *(0.928 )* | *(0.004 )* | *(0.332 )* |
|          | **Water** | *(0.949 )* | *( 0.002)* | *( 0.030)* |
| **DTR**  | **Oil**   | *( 0.896)* | *( o.112)* | *( 0.102)* |
|          | **Gas**   | *( 0.891)* | *( 0.086)* | *(0.120 )* |
|          | **Water** | *( 0.909)* | *(0.086 )* | *(0.111 )* |
| **RFR**  | **Oil**   | *(0.948 )* | *( 0.05)* | *(0.102 )* |
|          | **Gas**   | *( 0.944)* | *( 0.052)* | *(0.102 )* |
|          | **Water** | *( 0.947)* | *(0.06 )* | *( 0.052)* |



The results indicate that **LSTM may perform better for sequential data**, while **Random Forest Regressor provides robust performance with tabular data**.

---

## 🚀 How to Run the Notebooks
1. **Clone this repository:**
   ```sh
   git clone https://github.com/your-repo.git
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the notebooks in Jupyter:**
   ```sh
   jupyter notebook
   ```

---

## 🏆 Author
**Sunusi Muhammad Ibrahim**  
📌 CTO at **EJAZTECH.AI**  
📌 Petroleum Engineering Student | AI Enthusiast | Researcher  

💡 *Specializing in AI applications for petroleum engineering and beyond!*  

🔗 [LinkedIn](https://www.linkedin.com/in/sunusi-muhammad-ibrahim/)  

---

## 📜 License
This project is open-source under the **MIT License**.
```

---

### 🔥 Key Enhancements in This README:
- 📌 **Clear, structured overview**  
- 📝 **Dataset details included**  
- 🔍 **Concise explanation of each model**  
- 🏆 **Professional formatting**  
- 🚀 **Easy-to-follow steps for running the notebooks**  
- 📊 **Comparison of model performance**  
- 🔗 **LinkedIn & author credit**  

 
