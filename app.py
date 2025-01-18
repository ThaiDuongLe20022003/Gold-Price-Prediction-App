# Import Libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Input, Dense, Dropout, Attention, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set Appearance as Wide mode
st.set_page_config(layout = "wide")

# Reading Dataset and Data Preprocessing
df = pd.read_csv('gold.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by = 'Date', ascending = True, inplace = True)
df.reset_index(drop = True, inplace = True)
df['Price'] = df['Price'].replace({',': ''}, regex = True).astype(float)
df['Open'] = df['Open'].replace({',': ''}, regex = True).astype(float)
df['High'] = df['High'].replace({',': ''}, regex = True).astype(float)
df['Low'] = df['Low'].replace({',': ''}, regex = True).astype(float)
df.drop(['Vol.'], axis = 1, inplace = True)
df['Change %'] = df['Change %'].str.replace('%', '').astype(float) / 100

# Streamlit title
st.title('Gold Price Prediction Based On Historical Data')
st.write("In this project, we implement two models: Long-Short Term Memory with Attention Mechanism and Gated Reccurent Unit.")
st.write("\nHere's the data from 1975 to now.")

# Visualizing Gold Price in History
plt.figure(figsize = (25, 8), dpi = 150)
plt.rcParams['axes.facecolor'] = 'lightgray'
plt.rc('axes', edgecolor = 'white')

plt.plot(df.Date, df.Price, color = 'blue', lw = 2)

plt.title('Gold Price in History', fontsize = 15)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Price', fontsize = 12)

plt.grid(color = 'white')

plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()

st.pyplot(plt)

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    'Choose a model to predict gold prices',
    ['LSTM+Attention', 'GRU', 'Two Models']
)

# Input from user
input_year = st.number_input(
    "Enter the starting year for prediction (minimum: 2015):", 
    min_value = 2015, 
    max_value = 2024, 
    value = 2020
)

# Prepare data
features = ['Open', 'High', 'Low', 'Change %', 'Price']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features])

# Sliding windows technique
window_size = 3
def create_multivariate_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size, : - 1])  # Use all columns except 'Price'
        y.append(data[i + window_size, - 1])    # 'Price' as target
    return np.array(X), np.array(y)

# Split data into training and testing sets
train_data = data_scaled[df['Date'].dt.year < input_year]
test_data = data_scaled[df['Date'].dt.year >= input_year]

X_train, y_train = create_multivariate_sliding_windows(train_data, window_size)
X_test, y_test = create_multivariate_sliding_windows(test_data, window_size)

st.write(
    f"\nWe cannot train on future data in time series data, so we consider the data from {input_year} to now for testing and everything else for training."
)

# Visualize training and testing data
train_dates = df[df['Date'].dt.year < input_year]['Date'][window_size:]
test_dates = df[df['Date'].dt.year >= input_year]['Date'][window_size:]

plt.figure(figsize = (25, 8), dpi = 150)
plt.rcParams['axes.facecolor'] = 'lightgray'
plt.rc('axes',edgecolor = 'white')

plt.plot(train_dates, scaler.inverse_transform(np.hstack((np.zeros((y_train.shape[0], len(features) - 1)), y_train.reshape(-1, 1))))[:, -1], label='Training Data', color='blue')
plt.plot(test_dates, scaler.inverse_transform(np.hstack((np.zeros((y_test.shape[0], len(features) - 1)), y_test.reshape(-1, 1))))[:, -1], label='Testing Data', color='red')

plt.title('Gold Price Training and Test Sets', fontsize = 15)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Price', fontsize = 12)

plt.legend(
    ['Training set', 'Test set'],
    loc = 'upper left',
    prop = {'size': 15},
    facecolor = 'white',
    edgecolor = 'black'
)

plt.grid(color = 'white')

plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()

st.pyplot(plt)

# Define Models
def build_lstm_attention_multivariate(window_size, num_features):
    input_layer = Input(shape = (window_size, num_features))
    lstm_out = LSTM(64, return_sequences = True)(input_layer)
    
    # First Attention Layer
    attention_1 = Attention()([lstm_out, lstm_out])
    lstm_out_2 = LSTM(64, return_sequences = True)(attention_1)
    
    # Second Attention Layer
    attention_2 = Attention()([lstm_out_2, lstm_out_2])
    flatten = Flatten()(attention_2)
    
    dense_1 = Dense(64, activation = 'relu')(flatten)
    dropout = Dropout(0.2)(dense_1)
    output_layer = Dense(1)(dropout)

    model = Model(inputs = input_layer, outputs = output_layer)
    model.compile(loss = 'mean_squared_error', optimizer = 'Nadam')
    model.summary()
    return model

def build_gru_multivariate(window_size, num_features):
    input_layer = Input(shape = (window_size, num_features))
    gru_out = GRU(64, return_sequences = False)(input_layer)

    dense_1 = Dense(64, activation = 'relu')(gru_out)
    dropout = Dropout(0.2)(dense_1)
    output_layer = Dense(1)(dropout)

    model = Model(inputs = input_layer, outputs = output_layer)
    model.compile(loss = 'mean_squared_error', optimizer = 'Nadam')
    model.summary()
    return model

# Callbacks
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True, verbose = 1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, min_lr = 1e-8, verbose = 1)

# Metrics Calculating Definition
def calculate_metrics(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mape, mae, rmse

if model_option == 'LSTM+Attention':
    # Train LSTM+Attention
    start_time_lstm = time.time()
    lstm_model = build_lstm_attention_multivariate(window_size, X_train.shape[2])
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_split = 0.1,
        epochs = 200,
        batch_size = 128,
        callbacks = [early_stopping, reduce_lr],
        verbose = 1
    )
    end_time_lstm = time.time()

    # Evaluate LSTM+Attention Model
    lstm_results = lstm_model.evaluate(X_test, y_test, verbose = 1)
    lstm_pred = lstm_model.predict(X_test)
    lstm_r2 = r2_score(y_test, lstm_pred)

    # Inverse Transform Predictions
    y_test_actual = scaler.inverse_transform(np.hstack((np.zeros((y_test.shape[0], len(features) - 1)), y_test.reshape(-1, 1))))[:, -1]
    lstm_pred_true = scaler.inverse_transform(np.hstack((np.zeros((lstm_pred.shape[0], len(features) - 1)), lstm_pred)))[:, -1]

    # Calculate Metrics
    lstm_mape, lstm_mae, lstm_rmse = calculate_metrics(y_test_actual, lstm_pred_true)

    # Print Results
    st.write("\nRESULTS")
    st.write(f"Loss: {lstm_results}")
    st.write(f"R² Score: {lstm_r2}")
    st.write(f"Mean Absolute Percentage Error: {lstm_mape} %")
    st.write(f"Test Accuracy: {100 - lstm_mape} %")
    st.write(f"Mean Absolute Error: {lstm_mae} USD")
    st.write(f"Root Mean Square Error: {lstm_rmse} USD")
    st.write(f"Training time: {end_time_lstm - start_time_lstm} seconds")

    # Plot the actual test data and predicted test data
    plt.figure(figsize = (25, 8), dpi = 150)
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rc('axes', edgecolor = 'white')

    plt.plot(df[df['Date'].dt.year >= input_year]['Date'][window_size:], y_test_actual, color = 'red', lw = 2, label = 'Actual')
    plt.plot(df[df['Date'].dt.year >= input_year]['Date'][window_size:], lstm_pred_true, color = 'yellow', lw = 2, label = 'LSTM+Attention')

    plt.title('Model Performance on Gold Price Prediction', fontsize = 15)
    plt.xlabel('Year', fontsize = 12)
    plt.ylabel('Price', fontsize = 12)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc = 'upper left', prop = {'size': 15}, facecolor = 'white', edgecolor = 'black')
    plt.grid(color = 'white')

    st.pyplot(plt)

    # Calculate Residuals
    lstm_residuals = y_test_actual - lstm_pred_true

    # Scatterplot: Residuals vs True Values
    plt.figure(figsize = (8, 5))

    # LSTM+Attention Scatterplot
    plt.scatter(y_test_actual, lstm_residuals, color='blue', alpha = 0.6, edgecolor = 'k')
    plt.axhline(0, color = 'red', linestyle = '--', linewidth = 1)
    plt.title("LSTM+Attention: Residuals vs True Values", fontsize = 16)
    plt.xlabel("True Values", fontsize = 14)
    plt.ylabel("Residuals", fontsize = 14)
    plt.grid(True, linestyle = '--', alpha = 0.6)
    plt.tight_layout()

    st.pyplot(plt)

    # Histogram of LSTM+Attention
    plt.figure(figsize = (8, 5))
    sns.histplot(lstm_residuals, kde = True, bins = 30)
    plt.title("LSTM+Attention: Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()

    st.pyplot(plt)

elif model_option == 'GRU':
    # Train GRU
    start_time_gru = time.time()
    gru_model = build_gru_multivariate(window_size, X_train.shape[2])
    gru_history = gru_model.fit(
        X_train, y_train,
        validation_split = 0.1,
        epochs = 200,
        batch_size = 128,
        callbacks = [early_stopping, reduce_lr],
        verbose = 1
    )
    end_time_gru = time.time()

    # Evaluate GRU Model
    gru_results = gru_model.evaluate(X_test, y_test, verbose = 1)
    gru_pred = gru_model.predict(X_test)
    gru_r2 = r2_score(y_test, gru_pred)

    # Inverse Transform Predictions
    y_test_actual = scaler.inverse_transform(np.hstack((np.zeros((y_test.shape[0], len(features) - 1)), y_test.reshape(-1, 1))))[:, -1]
    gru_pred_true = scaler.inverse_transform(np.hstack((np.zeros((gru_pred.shape[0], len(features) - 1)), gru_pred)))[:, -1]

    # Calculate Metrics
    gru_mape, gru_mae, gru_rmse = calculate_metrics(y_test_actual, gru_pred_true)

    # Print Results
    st.write("\nRESULTS")
    st.write(f"Loss: {gru_results}")
    st.write(f"R² Score: {gru_r2}")
    st.write(f"Mean Absolute Percentage Error: {gru_mape} %")
    st.write(f"Test Accuracy: {100 - gru_mape} %")
    st.write(f"Mean Absolute Error: {gru_mae} USD")
    st.write(f"Root Mean Square Error: {gru_rmse} USD")
    st.write(f"Training Time: {end_time_gru - start_time_gru} seconds")

    # Plot the actual test data and predicted test data
    plt.figure(figsize = (25, 8), dpi = 150)
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rc('axes', edgecolor = 'white')

    plt.plot(df[df['Date'].dt.year >= input_year]['Date'][window_size:], y_test_actual, color = 'red', lw = 2, label = 'Actual')
    plt.plot(df[df['Date'].dt.year >= input_year]['Date'][window_size:], gru_pred_true, color = 'blue', lw = 2, label = 'GRU')

    plt.title('Model Performance on Gold Price Prediction', fontsize = 15)
    plt.xlabel('Year', fontsize = 12)
    plt.ylabel('Price', fontsize = 12)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc = 'upper left', prop = {'size': 15}, facecolor = 'white', edgecolor = 'black')
    plt.grid(color = 'white')

    st.pyplot(plt)

    # Calculate Residuals
    gru_residuals = y_test_actual - gru_pred_true

    # Scatterplot: Residuals vs True Values
    plt.figure(figsize = (8, 5))

    # GRU Scatterplot
    plt.scatter(y_test_actual, gru_residuals, color='green', alpha = 0.6, edgecolor = 'k')
    plt.axhline(0, color = 'red', linestyle = '--', linewidth = 1)
    plt.title("LSTM+Attention: Residuals vs True Values", fontsize = 16)
    plt.xlabel("True Values", fontsize = 14)
    plt.ylabel("Residuals", fontsize = 14)
    plt.grid(True, linestyle = '--', alpha = 0.6)
    plt.tight_layout()

    st.pyplot(plt)

    # Histogram of GRU
    plt.figure(figsize = (8, 5))
    sns.histplot(gru_residuals, kde = True, bins = 30)
    plt.title("GRU: Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()

    st.pyplot(plt)

else:
    # Training Models
    start_time_lstm = time.time()
    lstm_model = build_lstm_attention_multivariate(window_size, X_train.shape[2])
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_split = 0.1,
        epochs = 200,
        batch_size = 128,
        callbacks = [early_stopping, reduce_lr],
        verbose = 1
    )
    end_time_lstm = time.time()

   
    start_time_gru = time.time()
    gru_model = build_gru_multivariate(window_size, X_train.shape[2])
    gru_history = gru_model.fit(
        X_train, y_train,
        validation_split = 0.1,
        epochs = 200,
        batch_size = 128,
        callbacks = [early_stopping, reduce_lr],
        verbose = 1
    )
    end_time_gru = time.time()

    # Evaluate Models
    lstm_results = lstm_model.evaluate(X_test, y_test, verbose = 1)
    lstm_pred = lstm_model.predict(X_test)
    lstm_r2 = r2_score(y_test, lstm_pred)

    gru_results = gru_model.evaluate(X_test, y_test, verbose = 1)
    gru_pred = gru_model.predict(X_test)
    gru_r2 = r2_score(y_test, gru_pred)

    # Inverse Transform Predictions
    y_test_actual = scaler.inverse_transform(np.hstack((np.zeros((y_test.shape[0], len(features) - 1)), y_test.reshape(-1, 1))))[:, -1]
    lstm_pred_true = scaler.inverse_transform(np.hstack((np.zeros((lstm_pred.shape[0], len(features) - 1)), lstm_pred)))[:, -1]
    gru_pred_true = scaler.inverse_transform(np.hstack((np.zeros((gru_pred.shape[0], len(features) - 1)), gru_pred)))[:, -1]

    # Calculate Metrics
    lstm_mape, lstm_mae, lstm_rmse = calculate_metrics(y_test_actual, lstm_pred_true)
    gru_mape, gru_mae, gru_rmse = calculate_metrics(y_test_actual, gru_pred_true)

    # Print Results
    st.write("\nRESULTS")
    st.write("- LSTM+Attention:")
    st.write(f"Loss: {lstm_results}")
    st.write(f"R² Score: {lstm_r2}")
    st.write(f"Mean Absolute Percentage Error: {lstm_mape} %")
    st.write(f"Test Accuracy: {100 - lstm_mape} %")
    st.write(f"Mean Absolute Error: {lstm_mae} USD")
    st.write(f"Root Mean Square Error: {lstm_rmse} USD")
    st.write(f"Training time: {end_time_lstm - start_time_lstm} seconds")

    st.write("\n- GRU:")
    st.write(f"Loss: {gru_results}")
    st.write(f"R² Score: {gru_r2}")
    st.write(f"Mean Absolute Percentage Error: {gru_mape} %")
    st.write(f"Test Accuracy: {100 - gru_mape} %")
    st.write(f"Mean Absolute Error: {gru_mae} USD")
    st.write(f"Root Mean Square Error: {gru_rmse} USD")
    st.write(f"Training Time: {end_time_gru - start_time_gru} seconds")

    # Plot the actual test data and predicted test data
    plt.figure(figsize = (25, 8), dpi = 150)
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rc('axes', edgecolor = 'white')

    plt.plot(df[df['Date'].dt.year >= input_year]['Date'][window_size:], y_test_actual, color = 'red', lw = 2, label = 'Actual')
    plt.plot(df[df['Date'].dt.year >= input_year]['Date'][window_size:], lstm_pred_true, color = 'yellow', lw = 2, label = 'LSTM+Attention')
    plt.plot(df[df['Date'].dt.year >= input_year]['Date'][window_size:], gru_pred_true, color = 'blue', lw = 2, label = 'GRU')

    plt.title('Model Performance on Gold Price Prediction', fontsize = 15)
    plt.xlabel('Year', fontsize = 12)
    plt.ylabel('Price', fontsize = 12)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc = 'upper left', prop = {'size': 15}, facecolor = 'white', edgecolor = 'black')
    plt.grid(color = 'white')

    st.pyplot(plt)

    # Calculate Residuals
    lstm_residuals = y_test_actual - lstm_pred_true
    gru_residuals = y_test_actual - gru_pred_true

    # Scatterplot: Residuals vs True Values
    plt.figure(figsize = (14, 6), dpi = 150)

    plt.subplot(1, 2, 1)
    plt.scatter(y_test_actual, lstm_residuals, color = 'blue', alpha = 0.6)
    plt.axhline(0, color = 'red', linestyle = '--', linewidth = 1)
    plt.title("LSTM+Attention: Residuals vs True Values", fontsize = 14)
    plt.xlabel("True Values", fontsize = 12)
    plt.ylabel("Residuals", fontsize = 12)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(y_test_actual, gru_residuals, color = 'green', alpha = 0.6)
    plt.axhline(0, color = 'red', linestyle = '--', linewidth = 1)
    plt.title("GRU: Residuals vs True Values", fontsize = 14)
    plt.xlabel("True Values", fontsize = 12)
    plt.ylabel("Residuals", fontsize = 12)
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt)

    # Histogram of Residuals Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram of LSTM+Attention
    sns.histplot(lstm_residuals, kde = True, bins = 30, ax = axes[0])
    axes[0].set_title("LSTM+Attention: Residual Distribution")
    axes[0].set_xlabel("Residuals")
    axes[0].set_ylabel("Frequency")

    # Histogram of GRU
    sns.histplot(gru_residuals, kde = True, bins = 30, ax = axes[1])
    axes[1].set_title("GRU: Residual Distribution")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    st.pyplot(fig)