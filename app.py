# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import RobustScaler
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
df['Change %'] = df['Change %'].str.replace('%', '').astype(float) / 100

# Create New Variables
df['H-L'] = df['High'] - df['Low'] 
df['C-O'] = df['Price'] - df['Open'] 
df['30 DAYS MA'] = df['Price'].rolling(window = 30).mean() 
df['30 DAYS STD DEV'] = df['Price'].rolling(window = 30).std() 

# Drop
df.drop(['Vol.', 'Open', 'High', 'Low'], axis = 1, inplace = True)
df.dropna(inplace = True)

# Streamlit title
st.title('Gold Price Prediction Based On Historical Data')
st.write("In this project, we integrate Attention mechanism in two models: Long-Short Term Memory and Gated Reccurent Unit for Time Series Forcasting.")
st.write("\nHere's the data from 1975 to now.")

# Visualizing Gold Price in History
plt.figure(figsize = (25, 8), dpi = 250)
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
    ['Two Models', 'LSTM', 'GRU']
)

# Input from user
input_year = st.number_input(
    "Enter the starting year for prediction (minimum: 2015, maximum: 2024):", 
    min_value = 2015, 
    max_value = 2024, 
    value = 2020
)

# Prepare data
features = ['Change %', 'H-L', 'C-O', '30 DAYS MA', '30 DAYS STD DEV']
target = 'Price'
scaler = RobustScaler()

# Split data
train_data = df[df['Date'].dt.year < input_year]
test_data = df[df['Date'].dt.year >= input_year]

# Fit the scaler on the training data for both features and target
scaler.fit(train_data[features + [target]])

# Transform both training and test data
train_scaled = scaler.transform(train_data[features + [target]])
test_scaled = scaler.transform(test_data[features + [target]])

# Sliding windows technique
window_size = 30

def create_multivariate_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size, : - 1])
        y.append(data[i + window_size, - 1])
    return np.array(X), np.array(y)

X_train, y_train = create_multivariate_sliding_windows(train_scaled, window_size)
X_test, y_test = create_multivariate_sliding_windows(test_scaled, window_size)

st.write(
    f"\nWe cannot train on future data in time series data, so we consider the data from {input_year} to now for testing and everything else for training."
)

# Visualize training and testing data
test_size = df[df['Date'].dt.year >= input_year].shape[0]

plt.figure(figsize = (25, 8), dpi = 250)
plt.rcParams['axes.facecolor'] = 'lightgray'
plt.rc('axes',edgecolor = 'white')

plt.plot(df.Date[:-test_size], df.Price[:-test_size], color = 'blue', lw = 2)
plt.plot(df.Date[-test_size:], df.Price[-test_size:], color = 'red', lw = 2)

plt.title('Gold Price Training and Test Sets', fontsize = 15)
plt.xlabel('Date', fontsize = 12)
plt.ylabel('Price', fontsize = 12)

plt.legend(['Training set', 'Test set'], loc = 'upper left', prop = {'size': 15})
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

def build_gru_attention_multivariate(window_size, num_features):
    input_layer = Input(shape = (window_size, num_features))

    # First GRU Layer
    gru_out = GRU(64, return_sequences = True)(input_layer)

    # First Attention Layer
    attention_1 = Attention()([gru_out, gru_out])

    # Second GRU Layer
    gru_out_2 = GRU(64, return_sequences = True)(attention_1)

    # Second Attention Layer
    attention_2 = Attention()([gru_out_2, gru_out_2])

    flatten = Flatten()(attention_2)
    dense_1 = Dense(64, activation = 'relu')(flatten)
    dropout = Dropout(0.2)(dense_1)
    output_layer = Dense(1)(dropout)

    model = Model(inputs = input_layer, outputs = output_layer)
    model.compile(loss = 'mean_squared_error', optimizer = 'Nadam')
    model.summary()
    return model 

# Callbacks
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True, verbose = 1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, min_lr = 1e-6, verbose = 1)

# Metrics Calculating Definition
def calculate_metrics(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae = mean_absolute_error(y_true, y_pred)
    return mape, mae

if model_option == 'LSTM':
    # Train LSTM
    lstm_model = build_lstm_attention_multivariate(window_size, X_train.shape[2])

    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_split = 0.1,
        epochs = 200,
        batch_size = 128,
        callbacks = [early_stopping, reduce_lr],
        verbose = 1
    )

    # Evaluate LSTM Model
    lstm_results = lstm_model.evaluate(X_test, y_test, verbose = 1)
    lstm_rmse = np.sqrt(lstm_results)
    lstm_pred = lstm_model.predict(X_test)
    lstm_r2 = r2_score(y_test, lstm_pred)

    # Inverse Transform Predictions
    y_test_actual = scaler.inverse_transform(np.hstack((np.zeros((y_test.shape[0], len(features))), y_test.reshape(-1, 1))))[:, -1]
    lstm_pred_true = scaler.inverse_transform(np.hstack((np.zeros((lstm_pred.shape[0], len(features))), lstm_pred)))[:, -1]

    # Calculate Metrics
    lstm_mape, lstm_mae = calculate_metrics(y_test_actual, lstm_pred_true)

    # Print Results
    st.write("\nRESULTS")
    st.write(f"Loss: {lstm_results}")
    st.write(f"R² Score: {lstm_r2}")
    st.write(f"Mean Absolute Percentage Error: {lstm_mape} %")
    st.write(f"Mean Absolute Error: {lstm_mae} USD")
    st.write(f"Root Mean Square Error: {lstm_rmse} USD")

    # Plot the actual test data and predicted test data
    plt.figure(figsize = (25, 8), dpi = 250)
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rc('axes', edgecolor = 'white')

    plt.plot(df[df['Date'].dt.year >= input_year]['Date'][window_size:], y_test_actual, color = 'red', lw = 2, label = 'Actual')
    plt.plot(df[df['Date'].dt.year >= input_year]['Date'][window_size:], lstm_pred_true, color = 'yellow', lw = 2, label = 'LSTM')

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

    # LSTM Scatterplot
    plt.scatter(y_test_actual, lstm_residuals, color='blue', alpha = 0.6, edgecolor = 'k')
    plt.axhline(0, color = 'red', linestyle = '--', linewidth = 1)
    plt.title("LSTM: Residuals vs True Values", fontsize = 16)
    plt.xlabel("True Values", fontsize = 14)
    plt.ylabel("Residuals", fontsize = 14)
    plt.grid(True, linestyle = '--', alpha = 0.6)
    plt.tight_layout()

    st.pyplot(plt)

    # Histogram of LSTM
    plt.figure(figsize = (8, 5))
    sns.histplot(lstm_residuals, kde = True, bins = 30)
    plt.title("LSTM: Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()

    st.pyplot(plt)

elif model_option == 'GRU':
    # Train GRU
    gru_model = build_gru_attention_multivariate(window_size, X_train.shape[2])

    gru_history = gru_model.fit(
        X_train, y_train,
        validation_split = 0.1,
        epochs = 200,
        batch_size = 128,
        callbacks = [early_stopping, reduce_lr],
        verbose = 1
    )

    # Evaluate GRU Model
    gru_results = gru_model.evaluate(X_test, y_test, verbose = 1)
    gru_rmse = np.sqrt(gru_results)
    gru_pred = gru_model.predict(X_test)
    gru_r2 = r2_score(y_test, gru_pred)

    # Inverse Transform Predictions
    y_test_actual = scaler.inverse_transform(np.hstack((np.zeros((y_test.shape[0], len(features))), y_test.reshape(-1, 1))))[:, -1]
    gru_pred_true = scaler.inverse_transform(np.hstack((np.zeros((gru_pred.shape[0], len(features))), gru_pred)))[:, -1]

    # Calculate Metrics
    gru_mape, gru_mae = calculate_metrics(y_test_actual, gru_pred_true)

    # Print Results
    st.write("\nRESULTS")
    st.write(f"Loss: {gru_results}")
    st.write(f"R² Score: {gru_r2}")
    st.write(f"Mean Absolute Percentage Error: {gru_mape} %")
    st.write(f"Mean Absolute Error: {gru_mae} USD")
    st.write(f"Root Mean Square Error: {gru_rmse} USD")

    # Plot the actual test data and predicted test data
    plt.figure(figsize = (25, 8), dpi = 250)
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
    plt.scatter(y_test_actual, gru_residuals, color = 'green', alpha = 0.6, edgecolor = 'k')
    plt.axhline(0, color = 'red', linestyle = '--', linewidth = 1)
    plt.title("GRU: Residuals vs True Values", fontsize = 16)
    plt.xlabel("True Values", fontsize = 14)
    plt.ylabel("Residuals", fontsize = 14)
    plt.grid(True, linestyle = '--', alpha = 0.6)
    plt.tight_layout()

    st.pyplot(plt)

    # Histogram of GRU
    plt.figure(figsize = (8, 5))
    sns.histplot(gru_residuals, kde = True, bins = 30)
    plt.title("GRU: Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()

    st.pyplot(plt)

else:
    # Training Models
    lstm_model = build_lstm_attention_multivariate(window_size, X_train.shape[2])

    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_split = 0.1,
        epochs = 200,
        batch_size = 128,
        callbacks = [early_stopping, reduce_lr],
        verbose = 1
    )

   
    gru_model = build_gru_attention_multivariate(window_size, X_train.shape[2])

    gru_history = gru_model.fit(
        X_train, y_train,
        validation_split = 0.1,
        epochs = 200,
        batch_size = 128,
        callbacks = [early_stopping, reduce_lr],
        verbose = 1
    )

    # Evaluate Models
    lstm_results = lstm_model.evaluate(X_test, y_test, verbose = 1)
    lstm_rmse = np.sqrt(lstm_results)
    lstm_pred = lstm_model.predict(X_test)
    lstm_r2 = r2_score(y_test, lstm_pred)

    gru_results = gru_model.evaluate(X_test, y_test, verbose = 1)
    gru_rmse = np.sqrt(gru_results)
    gru_pred = gru_model.predict(X_test)
    gru_r2 = r2_score(y_test, gru_pred)

    # Inverse Transform Predictions
    y_test_actual = scaler.inverse_transform(np.hstack((np.zeros((y_test.shape[0], len(features))), y_test.reshape(-1, 1))))[:, -1]
    lstm_pred_true = scaler.inverse_transform(np.hstack((np.zeros((lstm_pred.shape[0], len(features))), lstm_pred)))[:, -1]
    gru_pred_true = scaler.inverse_transform(np.hstack((np.zeros((gru_pred.shape[0], len(features))), gru_pred)))[:, -1]

    # Calculate Metrics
    lstm_mape, lstm_mae = calculate_metrics(y_test_actual, lstm_pred_true)
    gru_mape, gru_mae = calculate_metrics(y_test_actual, gru_pred_true)

    # Print Results
    st.write("\nRESULTS")
    st.write("- LSTM:")
    st.write(f"Loss: {lstm_results}")
    st.write(f"R² Score: {lstm_r2}")
    st.write(f"Mean Absolute Percentage Error: {lstm_mape} %")
    st.write(f"Mean Absolute Error: {lstm_mae} USD")
    st.write(f"Root Mean Square Error: {lstm_rmse} USD")

    st.write("\n- GRU:")
    st.write(f"Loss: {gru_results}")
    st.write(f"R² Score: {gru_r2}")
    st.write(f"Mean Absolute Percentage Error: {gru_mape} %")
    st.write(f"Mean Absolute Error: {gru_mae} USD")
    st.write(f"Root Mean Square Error: {gru_rmse} USD")

    # Plot the actual test data and predicted test data
    plt.figure(figsize = (25, 8), dpi = 250)
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rc('axes', edgecolor = 'white')

    plt.plot(df[df['Date'].dt.year >= input_year]['Date'][window_size:], y_test_actual, color = 'red', lw = 2, label = 'Actual')
    plt.plot(df[df['Date'].dt.year >= input_year]['Date'][window_size:], lstm_pred_true, color = 'yellow', lw = 2, label = 'LSTM')
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
    plt.figure(figsize = (14, 6), dpi = 250)

    plt.subplot(1, 2, 1)
    plt.scatter(y_test_actual, lstm_residuals, color = 'blue', alpha = 0.6)
    plt.axhline(0, color = 'red', linestyle = '--', linewidth = 1)
    plt.title("LSTM: Residuals vs True Values", fontsize = 14)
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
    fig, axes = plt.subplots(1, 2, figsize = (14, 6))

    # Histogram of LSTM+Attention
    sns.histplot(lstm_residuals, kde = True, bins = 30, ax = axes[0])
    axes[0].set_title("LSTM: Residuals Distribution")
    axes[0].set_xlabel("Residuals")
    axes[0].set_ylabel("Frequency")

    # Histogram of GRU
    sns.histplot(gru_residuals, kde = True, bins = 30, ax = axes[1])
    axes[1].set_title("GRU: Residuals Distribution")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    st.pyplot(fig)