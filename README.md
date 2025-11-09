# week-2
This project uses a Long Short-Term Memory (LSTM) neural network to forecast hourly household electricity consumption. The model learns from the past 24 hours of energy data to predict the next hour’s usage, improving accuracy through deep learning–based time-series analysi
# Deep Learning–Based Energy Consumption Forecasting (Week 2)

This project extends the Week 1 work on household energy forecasting by applying a **Long Short-Term Memory (LSTM)** neural network.  
The objective is to predict the **next one-hour household electricity consumption** based on historical usage patterns.

---

## Project Overview

In Week 1, the model used a tuned XGBoost regressor with engineered lag, rolling, and weather features.  
In Week 2, the focus shifts to **time-series sequence modeling** using LSTM networks, which are designed to learn long-term dependencies in data.

The LSTM model processes sequences of past 24 hours of energy consumption to forecast the following hour’s demand.

---

## Dataset

- **Household Power Consumption Dataset** – UCI Machine Learning Repository  
  [Download Link](https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip)

  Minute-level readings from a single household (2006–2010) were resampled to **hourly averages** in Week 1.  
  The cleaned dataset (`cleaned_energy_data.csv`) is reused in this notebook.

---

## Methodology

1. **Data Loading**  
   Load the preprocessed dataset (`cleaned_energy_data.csv`) created in Week 1.

2. **Normalization**  
   Apply Min-Max scaling for stable LSTM training.

3. **Sequence Creation**  
   Use past 24 hours of data as input to predict the next 1 hour value.

4. **Model Development**  
   - Architecture: 1 LSTM (64 units) → Dropout (0.2) → Dense (32, ReLU) → Output (1).  
   - Optimizer: Adam Loss: MSE  
   - Early stopping to prevent over-fitting.

5. **Evaluation**  
   Evaluate using MAE, RMSE, and MAPE.

---

## Results

| Metric | Description | Result (approx.) |
|---------|--------------|------------------|
| **MAE** | Mean Absolute Error | ~0.30 – 0.35 kW |
| **RMSE** | Root Mean Squared Error | ~0.45 – 0.50 kW |
| **MAPE** | Mean Absolute Percentage Error | ~20 – 30 % |

The LSTM captures short-term consumption trends effectively and outperforms the baseline persistence model.

---

## Visualization

The notebook provides a plot comparing **Actual vs Predicted** consumption for a sample window, showing the model’s ability to follow real usage fluctuations.

---

## How to Run

1. Place `cleaned_energy_data.csv` and `energy_forecast_LSTM.ipynb` in the same folder.  
2. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib scikit-learn tensorflow
